#!/usr/bin/env python3
"""Gamepad teleoperation for the Kinova Gen3 7-DOF, driven from THIS laptop.

Reads a USB/wireless gamepad plugged into the machine running this script and
streams velocity commands straight to the arm over the Kortex API -- the same
high-level twist / joint-speed / gripper services the Kinova-supplied
controller uses. ROS2, MoveIt and ros2_control are not involved at all.

WHY THIS EXISTS (diagnostic value)
----------------------------------
The controller wired to the arm's own base does not move it, while the same
controller works on another computer. This script is the clean bisection:

  * arm MOVES under this script -> the arm, its actuators and the twist
    pipeline are healthy. The fault is in the arm-side controller path (the
    base USB port, the Web-App "Controller" mapping/profile, or that the
    controller was never bound to a session). Fix it in the Web App at
    https://192.168.1.10 -> Configurations -> Controller.
  * arm DOES NOT move under this script either -> the fault is upstream of the
    input device: check for an active fault/e-stop (this script prints and
    clears faults at start), admittance/servoing mode, or protection zones.

Either way you get a working teleop out of it.

TRANSPORT / SERVOING
--------------------
High-level SERVOING_MODE (SINGLE_LEVEL_SERVOING). Unlike ``impedance.py`` this
does NOT take low-level torque control -- the arm keeps its own position
safety envelope, joint limits and protection zones. Commands are velocity, so
the arm stops the moment the deadman is released or the stream stalls.

NETWORK PREREQUISITE
--------------------
The arm is at 192.168.1.10 on its own Ethernet subnet with no DHCP server, so
this laptop needs a static address on that subnet before anything below works::

    nmcli connection modify "Wired connection 1" ipv4.method manual \
        ipv4.addresses 192.168.1.100/24 ipv4.gateway "" ipv4.dns ""
    nmcli connection down "Wired connection 1" && nmcli connection up "Wired connection 1"
    ping -c1 192.168.1.10       # must reply before running this script

Do NOT run this while ``robot.launch.py`` / the kortex ros2_control driver
owns the arm: that driver holds its own session and streams position commands,
and the two will fight. Stop the ROS2 stack first.

RUN (A) -- laptop cabled straight to the arm
--------------------------------------------
The Kortex API lives in the impedance venv, not system python::

    ~/.venvs/kortex_impedance/bin/python gamepad_teleop.py

    # verify the gamepad only -- no robot, no network needed
    ~/.venvs/kortex_impedance/bin/python gamepad_teleop.py --list-input
    # print the twists that WOULD be sent, still without connecting
    ~/.venvs/kortex_impedance/bin/python gamepad_teleop.py --dry-run

RUN (B) -- arm cabled to REAL-1, gamepad on the laptop
------------------------------------------------------
SSH forwards a terminal, not a device: sshing into REAL-1 does NOT make the
laptop's /dev/input/js0 appear there, and REAL-1 has no joystick of its own.
So split the script in two -- the laptop serves input, REAL-1 drives the arm --
and carry the link inside the SSH connection you already open. No firewall
change, no extra exposed port, and the SSH session encrypts and authenticates
the stream for us.

Terminal 1, on the LAPTOP -- stream the pad (no robot, no kortex_api needed,
so plain system python3 is fine)::

    python3 gamepad_teleop.py --serve-input

Terminal 2, on the LAPTOP -- open SSH with a REVERSE tunnel, so REAL-1's
localhost:9123 comes back out to the laptop's input server, then run the arm
half on the far end::

    ssh -R 9123:localhost:9123 kinova@10.12.140.145
    # now at the REAL-1 prompt:
    python3 ~/ros2_kortex_ws/src/Kinova_surgical_arm_UCR/surgical_arm_bringup/\
scripts/gamepad_teleop.py --input-net 9123

Both halves must be the same file -- keep them in sync when editing.

LINK WATCHDOG (why this is safe over Wi-Fi)
-------------------------------------------
Measured laptop->REAL-1 RTT is 19-195 ms with heavy jitter, so the link is
assumed to be unreliable. Frames are absolute state snapshots, never deltas, so
a dropped frame cannot desynchronise anything. If the newest frame goes older
than --input-timeout (0.30 s), the arm half zeroes every control and holds the
arm stopped, exactly as if the deadman had been released -- covering a Wi-Fi
stall, a closed laptop lid, a dropped tunnel, or a killed server. The link
recovers on its own without dropping the Kortex session.

CONTROLS (Logitech F710, switch on the back set to "X" / XInput)
----------------------------------------------------------------
Nothing moves unless the deadman is held.

    RB (hold)            DEADMAN -- required for ALL motion
    LB (hold)            rotation modifier (see below)

  cartesian mode (default)
    left stick  Y/X      linear  +X / +Y   (base frame, m/s)
    right stick Y        linear  +Z
    with LB held:
    left stick  Y/X      angular wx / wy   (deg/s)
    right stick X        angular wz

  joint mode (press X to toggle)
    D-pad left/right     select joint 1..7
    left stick Y         drive the selected joint (deg/s)

  always
    LT / RT              close / open the gripper
    D-pad up/down        speed scale up / down (10% steps)
    A                    hard stop (zero everything, calls Base.Stop)
    Y                    clear faults
    BACK                 quit (releases the session cleanly)

A DirectInput ("D") F710, an Xbox pad, or anything else enumerating on
/dev/input/js* works too -- run ``--list-input``, read off the axis/button
numbers, and pass ``--mapping`` with the ones that differ, e.g.
``--mapping deadman=4,lin_x=1,trigger_left=2``.
"""
import argparse
import array
import fcntl
import json
import os
import socket
import struct
import sys
import threading
import time

# --- protobuf 3.5.1 (shipped with kortex_api 2.6.0) compat shim for Py>=3.10 --
# Same shim as impedance.py: protobuf 3.5.1's containers.py references the
# collections.MutableMapping aliases moved to collections.abc in 3.10. Must run
# BEFORE anything that imports protobuf.
import collections
import collections.abc
for _name in ("MutableMapping", "Mapping", "Sequence", "MutableSequence",
              "Callable", "Iterable"):
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(collections.abc, _name))

# The Kortex API is only needed on the machine that actually talks to the arm.
# --serve-input (the laptop half of a split setup), --list-input and --dry-run
# are pure input paths, so a missing kortex_api must not stop them: import it
# softly and fail only when a real connection is attempted.
try:
    from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
    from kortex_api.SessionManager import SessionManager
    from kortex_api.TCPTransport import TCPTransport
    from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
    from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
    from kortex_api.autogen.messages import Base_pb2, Session_pb2
    KORTEX_IMPORT_ERROR = None
except ImportError as _exc:      # noqa: F841
    KORTEX_IMPORT_ERROR = str(_exc)

# --- Connection --------------------------------------------------------------
TCP_PORT = 10000
DEFAULT_IP = "192.168.1.10"
DEFAULT_CREDENTIALS = ("admin", "admin")

# --- Command rate ------------------------------------------------------------
# High-level twist wants a steady stream; the arm decelerates on its own if the
# stream stops. 40 Hz is smooth and leaves plenty of TCP headroom. Each command
# carries duration=0 ("until superseded"), so a stalled loop coasts to a stop
# rather than latching the last velocity forever.
DEFAULT_RATE_HZ = 40.0

# --- Speed limits (full-stick deflection at scale 1.0) -----------------------
# Deliberately gentle. These are teleop speeds for a 1.2 m arm within arm's
# reach of a person; raise with --max-lin / --max-ang / --max-joint if the
# motion feels sluggish, but do it in small steps.
MAX_LINEAR_M_S = 0.10        # m/s   -- Kortex twist linear units
MAX_ANGULAR_DEG_S = 20.0     # deg/s -- Kortex twist angular units
MAX_JOINT_DEG_S = 15.0       # deg/s -- Kortex joint-speed units

# Stick deadzone as a fraction of full scale. F710 sticks rest around +/-2 %.
DEADZONE = 0.12
# Trigger threshold: F710 triggers rest at -1.0 and go to +1.0 fully pressed.
TRIGGER_ON = 0.5

# Gripper is commanded in SPEED mode; sign selects direction, magnitude is
# fraction of max gripper speed. Positive closes on the Robotiq 2F.
GRIPPER_SPEED = 0.6

# --- Gamepad profiles --------------------------------------------------------
# The F710's rear switch picks between two COMPLETELY different enumerations,
# and getting them confused is a safety problem, not a nuisance: in DirectInput
# the XInput trigger axes (2, 5) land on the right stick and the D-pad, so
# nudging the right stick would work the GRIPPER, and XInput's BACK (button 6)
# is DirectInput's LT, so a trigger pull would quit mid-motion. Ship both
# layouts and detect which one is plugged in rather than trusting the switch.
#
#   X (XInput)      -> "Logitech Gamepad F710",            8 axes, 11 buttons
#   D (DirectInput) -> "Logitech Cordless RumblePad 2",    6 axes, 12 buttons
#
# DirectInput has no analog triggers -- LT/RT are plain buttons -- so each
# profile declares how to read them.
XINPUT_MAPPING = {
    # axes
    "lin_x": 1,          # left stick Y  (pushed forward = -1 -> inverted below)
    "lin_y": 0,          # left stick X
    "lin_z": 4,          # right stick Y (inverted below)
    "ang_z": 3,          # right stick X
    "trigger_left": 2,   # LT (analog, rests at -1)
    "trigger_right": 5,  # RT (analog, rests at -1)
    "dpad_x": 6,
    "dpad_y": 7,
    # buttons
    "deadman": 5,        # RB
    "rotate_mod": 4,     # LB
    "toggle_joint": 2,   # X
    "stop": 0,           # A
    "clear_faults": 3,   # Y
    "quit": 6,           # BACK
}

DINPUT_MAPPING = {
    # axes
    "lin_x": 1,          # left stick Y
    "lin_y": 0,          # left stick X
    "lin_z": 3,          # right stick Y
    "ang_z": 2,          # right stick X
    "trigger_left": 6,   # LT -- a BUTTON in this mode
    "trigger_right": 7,  # RT -- a BUTTON in this mode
    "dpad_x": 4,
    "dpad_y": 5,
    # buttons
    "deadman": 5,        # RB
    "rotate_mod": 4,     # LB
    "toggle_joint": 0,   # X
    "stop": 1,           # A
    "clear_faults": 3,   # Y
    "quit": 8,           # BACK
}

PROFILES = {
    "xinput": {"mapping": XINPUT_MAPPING, "analog_triggers": True,
               "label": "XInput (rear switch on X)"},
    "dinput": {"mapping": DINPUT_MAPPING, "analog_triggers": False,
               "label": "DirectInput (rear switch on D)"},
}

# Axes whose raw sign is opposite the intended robot direction. Sticks report
# -1 when pushed forward/up, and we want forward/up to be positive. Same in
# both profiles -- only the axis NUMBERS move.
INVERTED_AXES = ("lin_x", "lin_z", "dpad_y")


def detect_profile(name, n_axes, n_buttons):
    """Pick a profile from what the device actually reports.

    Name first (unambiguous for the two known modes), then the axis/button
    counts as a fallback for relabelled clones. Returns (profile_name, why).
    """
    low = (name or "").lower()
    if "rumblepad" in low or "dual action" in low:
        return "dinput", f"device name {name!r}"
    if "gamepad f710" in low or "x-box" in low or "xbox" in low:
        return "xinput", f"device name {name!r}"
    if (n_axes, n_buttons) == (6, 12):
        return "dinput", f"{n_axes} axes / {n_buttons} buttons"
    if (n_axes, n_buttons) == (8, 11):
        return "xinput", f"{n_axes} axes / {n_buttons} buttons"
    return "xinput", (f"unrecognised pad ({name!r}, {n_axes} axes, "
                      f"{n_buttons} buttons) -- ASSUMING XInput; verify with "
                      f"--list-input before trusting the controls")

# --- Linux joystick API ------------------------------------------------------
# struct js_event { __u32 time; __s16 value; __u8 type; __u8 number; }
JS_EVENT_FMT = "IhBB"
JS_EVENT_SIZE = struct.calcsize(JS_EVENT_FMT)
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80
JSIOCGNAME_LEN = 128


def _jsiocgname(fd):
    """JSIOCGNAME(len) -- read the device name string from an open js fd."""
    buf = array.array("B", [0] * JSIOCGNAME_LEN)
    # _IOC(_IOC_READ, 'j', 0x13, len) == 0x80006A13 | (len << 16)
    ioctl_op = 0x80006A13 + (JSIOCGNAME_LEN << 16)
    try:
        fcntl.ioctl(fd, ioctl_op, buf)
    except OSError:
        return "unknown"
    return buf.tobytes().rstrip(b"\x00").decode("utf-8", "replace")


def _jsioc_count(fd, op):
    """JSIOCGAXES / JSIOCGBUTTONS -- how many the device reports."""
    buf = array.array("B", [0])
    try:
        fcntl.ioctl(fd, op, buf)
    except OSError:
        return 0
    return buf[0]


class PadState:
    """Axis/button state plus rising-edge tracking, shared by both sources."""

    def __init__(self, name="pad"):
        self.name = name
        self.profile = "xinput"   # replaced by detection / by the stream
        self.profile_why = "default"
        self.axes = {}       # number -> float in [-1, 1]
        self.buttons = {}    # number -> 0/1
        self._prev_buttons = {}

    def axis(self, number):
        return self.axes.get(number, 0.0)

    def button(self, number):
        return bool(self.buttons.get(number, 0))

    def pressed(self, number):
        """True on the rising edge only -- call once per loop per button."""
        now = bool(self.buttons.get(number, 0))
        was = self._prev_buttons.get(number, False)
        self._prev_buttons[number] = now
        return now and not was

    def zero(self):
        """Force every control to neutral (used when input goes stale)."""
        for k in self.axes:
            self.axes[k] = 0.0
        for k in self.buttons:
            self.buttons[k] = 0


class Gamepad(PadState):
    """Non-blocking reader for a /dev/input/js* device.

    Uses the raw joystick protocol rather than pygame/evdev so the script has
    no dependency beyond the Kortex API -- the impedance venv has neither
    input library installed, and adding one there for a teleop script is not
    worth the risk to the impedance runs.
    """

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        self.name = _jsiocgname(self.fd)
        self.n_axes = _jsioc_count(self.fd, 0x80016A11)      # JSIOCGAXES
        self.n_buttons = _jsioc_count(self.fd, 0x80016A12)   # JSIOCGBUTTONS
        self.profile, self.profile_why = detect_profile(
            self.name, self.n_axes, self.n_buttons)

    def close(self):
        try:
            os.close(self.fd)
        except OSError:
            pass

    def poll(self):
        """Drain pending events into self.axes / self.buttons.

        Returns False if the device disappeared (unplugged / powered off),
        which the caller must treat as a stop condition.
        """
        while True:
            try:
                data = os.read(self.fd, JS_EVENT_SIZE)
            except BlockingIOError:
                return True
            except OSError:
                return False
            if not data or len(data) < JS_EVENT_SIZE:
                return True
            _, value, ev_type, number = struct.unpack(JS_EVENT_FMT, data)
            # INIT events are synthetic startup values; they carry real state,
            # so fold them in but strip the flag.
            ev_type &= ~JS_EVENT_INIT
            if ev_type == JS_EVENT_AXIS:
                self.axes[number] = max(-1.0, value / 32767.0)
            elif ev_type == JS_EVENT_BUTTON:
                self.buttons[number] = value


# --- Split operation: pad on the laptop, arm on REAL-1 -----------------------
# The robot is cabled to REAL-1 but the gamepad is on the laptop, and SSH
# forwards a terminal, not a device -- /dev/input/js0 does not appear on the
# far end. So the laptop runs --serve-input (pure input, no Kortex, no robot)
# and REAL-1 runs --input-net, which drives the arm from the streamed state.
#
# The link is a plain TCP stream of newline-delimited JSON frames, one per
# input tick, carrying the full axis/button snapshot (not deltas -- a dropped
# frame must not desynchronise the state). It is meant to be tunnelled inside
# SSH, which supplies the encryption and the authentication; the server
# therefore binds to loopback only by default and never accepts a second
# client while one is connected.
INPUT_PROTOCOL_VERSION = 1
DEFAULT_INPUT_PORT = 9123
# If the newest frame is older than this, the far end has gone quiet (Wi-Fi
# stall, laptop asleep, SSH tunnel dropped, script killed). Treat it exactly
# like a released deadman and zero the arm. Measured RTT to REAL-1 is 19-195 ms
# with heavy jitter, so this must clear the worst case with margin while still
# stopping the arm well inside a hand's reaction time.
DEFAULT_INPUT_TIMEOUT = 0.30


class InputServer:
    """Laptop side: read the local pad, stream snapshots to one TCP client."""

    def __init__(self, pad, host, port, rate_hz):
        self.pad = pad
        self.host = host
        self.port = port
        self.period = 1.0 / rate_hz
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(1)

    def serve_forever(self):
        seq = 0
        while True:
            print(f"waiting for a client on {self.host}:{self.port} ...")
            conn, peer = self.sock.accept()
            # Without TCP_NODELAY, Nagle coalesces these tiny frames and adds
            # tens of ms of lag to every stick movement.
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"client connected from {peer[0]}:{peer[1]}")
            try:
                while True:
                    tick = time.time()
                    if not self.pad.poll():
                        print("\ngamepad disconnected -- closing client link")
                        # Send one neutral frame so the far end stops on a
                        # clean zero rather than waiting out the timeout.
                        self.pad.zero()
                        self._send(conn, seq, tick)
                        return
                    seq += 1
                    self._send(conn, seq, tick)
                    sys.stdout.write(f"\rstreaming frame {seq}   ")
                    sys.stdout.flush()
                    sleep = self.period - (time.time() - tick)
                    if sleep > 0:
                        time.sleep(sleep)
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                print(f"\nclient disconnected ({exc}); waiting for a new one")
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    def _send(self, conn, seq, tick):
        frame = {
            "v": INPUT_PROTOCOL_VERSION,
            "seq": seq,
            "t": tick,
            "name": self.pad.name,
            # The serving machine is the only one that can see the device, so
            # it is the authority on which layout is plugged in.
            "profile": self.pad.profile,
            # JSON object keys must be strings; the far end casts back to int.
            "axes": {str(k): round(v, 4) for k, v in self.pad.axes.items()},
            "buttons": {str(k): int(v) for k, v in self.pad.buttons.items()},
        }
        conn.sendall((json.dumps(frame) + "\n").encode("utf-8"))

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class NetGamepad(PadState):
    """REAL-1 side: same interface as Gamepad, fed by an InputServer stream.

    A background thread owns the socket so a slow or stalled link can never
    block the command loop -- the loop just sees stale state, and staleness
    is what trips the watchdog.
    """

    def __init__(self, host, port, timeout=DEFAULT_INPUT_TIMEOUT,
                 connect_timeout=10.0):
        super().__init__(name="remote pad")
        self.timeout = timeout
        self.sock = socket.create_connection((host, port), connect_timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(1.0)
        self.last_frame_t = 0.0
        self.frames = 0
        self.link_up = True
        self._alive = True
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        buf = b""
        while self._alive:
            try:
                chunk = self.sock.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, _, buf = buf.partition(b"\n")
                if not line.strip():
                    continue
                try:
                    frame = json.loads(line)
                except ValueError:
                    continue
                if frame.get("v") != INPUT_PROTOCOL_VERSION:
                    continue
                axes = {int(k): float(v)
                        for k, v in frame.get("axes", {}).items()}
                buttons = {int(k): int(v)
                           for k, v in frame.get("buttons", {}).items()}
                with self._lock:
                    self.axes.update(axes)
                    self.buttons.update(buttons)
                    self.last_frame_t = time.time()
                    self.frames += 1
                    if frame.get("name"):
                        self.name = frame["name"]
                    if frame.get("profile") in PROFILES:
                        self.profile = frame["profile"]
                        self.profile_why = "reported by the serving machine"
        self._alive = False

    def poll(self):
        """Returns False once the link is gone for good.

        A merely STALE link is not fatal -- it zeroes the controls (so the arm
        stops) but keeps running, so a Wi-Fi hiccup does not tear down the
        Kortex session and force a restart.
        """
        with self._lock:
            age = time.time() - self.last_frame_t if self.last_frame_t else 1e9
            stale = age > self.timeout
            if stale:
                self.zero()
        if stale and self.link_up:
            self.link_up = False
            print(f"\n[link] input stale ({age:.2f}s) -- arm held stopped")
        elif not stale and not self.link_up:
            self.link_up = True
            print("\n[link] input recovered")
        return self._alive or not stale

    def wait_for_first_frame(self, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self.frames:
                    return True
            if not self._alive:
                return False
            time.sleep(0.05)
        return False

    def close(self):
        self._alive = False
        try:
            self.sock.close()
        except OSError:
            pass


def apply_deadzone(value, deadzone=DEADZONE):
    """Rescale so output starts at 0 at the deadzone edge (no step at break-out)."""
    if abs(value) < deadzone:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


class ArmConnection:
    """Kortex session + high-level velocity command helpers."""

    def __init__(self, ip, username, password, dry_run=False):
        self.dry_run = dry_run
        self.base = None
        self.actuator_count = 7
        self._transport = None
        self._session = None
        self._router = None
        if dry_run:
            return
        if KORTEX_IMPORT_ERROR is not None:
            raise SystemExit(
                f"the Kortex API is not importable here ({KORTEX_IMPORT_ERROR}).\n"
                "Run the arm half on the machine cabled to the robot, with an\n"
                "interpreter that has kortex_api (on this laptop that is\n"
                "~/.venvs/kortex_impedance/bin/python).")

        self._transport = TCPTransport()
        self._router = RouterClient(self._transport,
                                    lambda kex: print(f"[router] {kex}"))
        self._transport.connect(ip, TCP_PORT)

        info = Session_pb2.CreateSessionInfo()
        info.username = username
        info.password = password
        info.session_inactivity_timeout = 60000     # ms
        info.connection_inactivity_timeout = 2000   # ms
        self._session = SessionManager(self._router)
        self._session.CreateSession(info)

        self.base = BaseClient(self._router)
        self.base_cyclic = BaseCyclicClient(self._router)
        self.actuator_count = self.base.GetActuatorCount().count

        # High-level servoing: the arm keeps its own limits and safety envelope.
        servo = Base_pb2.ServoingModeInformation()
        servo.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(servo)

        # Non-blocking send options: a teleop loop must never stall waiting on
        # an ack, or the command stream stutters and the arm jerks.
        self._opts = RouterClientSendOptions()
        self._opts.timeout_ms = 100

    def report_faults(self):
        """Print anything the arm is currently complaining about."""
        if self.dry_run:
            return
        try:
            fb = self.base_cyclic.RefreshFeedback()
        except Exception as exc:                      # noqa: BLE001
            print(f"[faults] could not read feedback: {exc}")
            return
        flags = []
        if fb.base.fault_bank_a or fb.base.fault_bank_b:
            flags.append(f"base fault banks: A=0x{fb.base.fault_bank_a:08x} "
                         f"B=0x{fb.base.fault_bank_b:08x}")
        for i, act in enumerate(fb.actuators):
            if act.fault_bank_a or act.fault_bank_b:
                flags.append(f"actuator {i+1}: A=0x{act.fault_bank_a:08x} "
                             f"B=0x{act.fault_bank_b:08x}")
        if flags:
            print("[faults] ACTIVE:")
            for f in flags:
                print(f"         {f}")
            print("         press Y to clear, or use the Web App if they persist")
        else:
            print("[faults] none active")

    def clear_faults(self):
        if self.dry_run:
            print("[dry-run] ClearFaults()")
            return
        try:
            self.base.ClearFaults()
            print("[faults] cleared")
        except Exception as exc:                      # noqa: BLE001
            print(f"[faults] clear failed: {exc}")

    def send_twist(self, lin, ang, frame_base=True):
        """lin = (x,y,z) m/s, ang = (wx,wy,wz) deg/s."""
        if self.dry_run:
            return
        cmd = Base_pb2.TwistCommand()
        cmd.reference_frame = (Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
                               if frame_base else
                               Base_pb2.CARTESIAN_REFERENCE_FRAME_TOOL)
        cmd.duration = 0     # apply until superseded; stream keeps it alive
        cmd.twist.linear_x, cmd.twist.linear_y, cmd.twist.linear_z = lin
        cmd.twist.angular_x, cmd.twist.angular_y, cmd.twist.angular_z = ang
        self.base.SendTwistCommand(cmd, options=self._opts)

    def send_joint_speeds(self, speeds_deg_s):
        """speeds_deg_s: list of length actuator_count, deg/s."""
        if self.dry_run:
            return
        cmd = Base_pb2.JointSpeeds()
        for i, v in enumerate(speeds_deg_s):
            js = cmd.joint_speeds.add()
            js.joint_identifier = i
            js.value = v
            js.duration = 0
        self.base.SendJointSpeedsCommand(cmd, options=self._opts)

    def send_gripper_speed(self, value):
        """value in [-1, 1]; positive closes, 0 stops."""
        if self.dry_run:
            return
        cmd = Base_pb2.GripperCommand()
        cmd.mode = Base_pb2.GRIPPER_SPEED
        finger = cmd.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = value
        self.base.SendGripperCommand(cmd, options=self._opts)

    def stop(self):
        """Zero every command channel, then ask the arm to stop outright."""
        if self.dry_run:
            print("[dry-run] stop()")
            return
        for attempt in (self._zero_twist, self._zero_gripper, self._base_stop):
            try:
                attempt()
            except Exception as exc:                  # noqa: BLE001
                print(f"[stop] {attempt.__name__} failed: {exc}")

    def _zero_twist(self):
        self.send_twist((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    def _zero_gripper(self):
        self.send_gripper_speed(0.0)

    def _base_stop(self):
        self.base.Stop()

    def close(self):
        if self.dry_run:
            return
        try:
            self._session.CloseSession()
        except Exception:                             # noqa: BLE001
            pass
        try:
            self._router.SetActivationStatus(False)
        except Exception:                             # noqa: BLE001
            pass
        try:
            self._transport.disconnect()
        except Exception:                             # noqa: BLE001
            pass


def parse_mapping_overrides(text, mapping):
    """--mapping "deadman=4,lin_x=1" -> mutated copy of mapping."""
    out = dict(mapping)
    if not text:
        return out
    for pair in text.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise SystemExit(f"--mapping: expected name=number, got '{pair}'")
        name, _, num = pair.partition("=")
        name = name.strip()
        if name not in out:
            raise SystemExit(
                f"--mapping: unknown control '{name}'.\n"
                f"known: {', '.join(sorted(out))}")
        try:
            out[name] = int(num)
        except ValueError:
            raise SystemExit(f"--mapping: '{num}' is not an integer")
    return out


def list_input(path):
    """Print raw axis/button activity so an unknown pad can be mapped."""
    pad = Gamepad(path)
    print(f"reading {path}  ({pad.name})")
    print(f"{pad.n_axes} axes, {pad.n_buttons} buttons -> "
          f"{PROFILES[pad.profile]['label']}  [{pad.profile_why}]")
    print("move each stick and press each button; Ctrl-C to stop\n")
    last = {}
    try:
        while True:
            if not pad.poll():
                print("device disappeared")
                return 1
            for num, val in sorted(pad.axes.items()):
                if abs(val - last.get(("a", num), 0.0)) > 0.15:
                    last[("a", num)] = val
                    print(f"  axis   {num:2d}  {val:+.2f}")
            for num, val in sorted(pad.buttons.items()):
                if val != last.get(("b", num), 0):
                    last[("b", num)] = val
                    if val:
                        print(f"  button {num:2d}  pressed")
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\ndone")
        return 0
    finally:
        pad.close()


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Gamepad teleop for the Kinova Gen3 over the Kortex API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("CONTROLS")[1] if "CONTROLS" in __doc__ else None)
    p.add_argument("--robot-ip", default=DEFAULT_IP)
    p.add_argument("--username", default=DEFAULT_CREDENTIALS[0])
    p.add_argument("--password", default=DEFAULT_CREDENTIALS[1])
    p.add_argument("--device", default="/dev/input/js0",
                   help="joystick device (default: %(default)s)")
    p.add_argument("--rate", type=float, default=DEFAULT_RATE_HZ,
                   help="command rate in Hz (default: %(default)s)")
    p.add_argument("--max-lin", type=float, default=MAX_LINEAR_M_S,
                   help="linear speed at full stick, m/s (default: %(default)s)")
    p.add_argument("--max-ang", type=float, default=MAX_ANGULAR_DEG_S,
                   help="angular speed at full stick, deg/s (default: %(default)s)")
    p.add_argument("--max-joint", type=float, default=MAX_JOINT_DEG_S,
                   help="joint speed at full stick, deg/s (default: %(default)s)")
    p.add_argument("--scale", type=float, default=0.5,
                   help="initial speed scale 0..1, D-pad adjusts (default: %(default)s)")
    p.add_argument("--tool-frame", action="store_true",
                   help="interpret cartesian sticks in the TOOL frame instead of BASE")
    p.add_argument("--no-deadman", action="store_true",
                   help="UNSAFE: allow motion without holding RB")
    p.add_argument("--pad-profile", choices=("auto", "xinput", "dinput"),
                   default="auto",
                   help="controller layout; 'auto' detects it from the device "
                        "(default: %(default)s)")
    p.add_argument("--mapping", default=None,
                   help="override axis/button numbers on top of the profile, "
                        "e.g. 'deadman=4,lin_x=1'")
    p.add_argument("--list-input", action="store_true",
                   help="print raw gamepad events and exit (no robot connection)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the commands that would be sent; never connects")

    split = p.add_argument_group(
        "split operation (pad on the laptop, arm on another machine)")
    split.add_argument("--serve-input", action="store_true",
                       help="LAPTOP half: read the local pad and stream it; "
                            "never touches the robot and needs no kortex_api")
    split.add_argument("--input-net", metavar="[HOST:]PORT",
                       help="ROBOT half: take input from a --serve-input "
                            "stream instead of a local joystick")
    split.add_argument("--input-port", type=int, default=DEFAULT_INPUT_PORT,
                       help="port for --serve-input (default: %(default)s)")
    split.add_argument("--input-bind", default="127.0.0.1",
                       help="bind address for --serve-input; loopback by "
                            "default because the link belongs inside an SSH "
                            "tunnel (default: %(default)s)")
    split.add_argument("--input-rate", type=float, default=DEFAULT_RATE_HZ,
                       help="frames/s sent by --serve-input (default: %(default)s)")
    split.add_argument("--input-timeout", type=float,
                       default=DEFAULT_INPUT_TIMEOUT,
                       help="stop the arm if no input frame arrives for this "
                            "many seconds (default: %(default)s)")
    args = p.parse_args(argv)

    if args.serve_input and args.input_net:
        p.error("--serve-input and --input-net are the two OPPOSITE halves of "
                "a split setup; pass one or the other, on the two machines.")

    # --input-net takes its input from the network, so a local joystick is
    # neither needed nor looked for.
    if args.input_net:
        return run_input_client(args, p)

    if not os.path.exists(args.device):
        print(f"error: no joystick at {args.device}", file=sys.stderr)
        print("       plug the pad in (F710: switch on the back set to 'X'),",
              file=sys.stderr)
        print("       then check:  ls -l /dev/input/js*", file=sys.stderr)
        return 2

    if args.list_input:
        return list_input(args.device)

    pad = Gamepad(args.device)
    print(f"gamepad: {pad.name}  ({args.device}, "
          f"{pad.n_axes} axes, {pad.n_buttons} buttons)")

    if args.serve_input:
        return run_input_server(args, pad)

    return run_teleop(args, pad)


def run_input_server(args, pad):
    """LAPTOP half of a split setup: stream the local pad, drive nothing."""
    try:
        server = InputServer(pad, args.input_bind, args.input_port,
                             args.input_rate)
    except OSError as exc:
        pad.close()
        print(f"error: cannot bind {args.input_bind}:{args.input_port}: {exc}",
              file=sys.stderr)
        return 4
    print(f"input server on {args.input_bind}:{args.input_port} "
          f"at {args.input_rate:.0f} Hz -- the robot half connects to this.")
    print("this process NEVER touches the robot; Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nCtrl-C -- input server stopping")
    finally:
        server.close()
        pad.close()
    return 0


def run_input_client(args, parser):
    """ROBOT half of a split setup: drive the arm from a streamed pad."""
    spec = args.input_net
    if ":" in spec:
        host, _, port_s = spec.rpartition(":")
    else:
        host, port_s = "127.0.0.1", spec
    try:
        port = int(port_s)
    except ValueError:
        parser.error(f"--input-net: '{spec}' is not [HOST:]PORT")

    print(f"connecting to input stream at {host}:{port} ...")
    try:
        pad = NetGamepad(host, port, timeout=args.input_timeout)
    except OSError as exc:
        print(f"error: no input stream at {host}:{port}: {exc}", file=sys.stderr)
        print("       start the laptop half first:", file=sys.stderr)
        print("         gamepad_teleop.py --serve-input", file=sys.stderr)
        print("       and make sure the SSH reverse tunnel is up:", file=sys.stderr)
        print(f"         ssh -R {port}:localhost:{port} kinova@10.12.140.145",
              file=sys.stderr)
        return 4

    if not pad.wait_for_first_frame():
        pad.close()
        print("error: connected, but no input frames arrived. Is the gamepad",
              file=sys.stderr)
        print("       actually attached on the serving machine?", file=sys.stderr)
        return 4
    print(f"input stream up: {pad.name}  "
          f"(watchdog {args.input_timeout:.2f}s)")
    return run_teleop(args, pad)


def run_teleop(args, pad):
    """The command loop. Identical whether the pad is local or streamed."""
    if args.pad_profile == "auto":
        profile_name, why = pad.profile, pad.profile_why
    else:
        profile_name, why = args.pad_profile, "forced with --pad-profile"
    profile = PROFILES[profile_name]
    analog_triggers = profile["analog_triggers"]
    mapping = parse_mapping_overrides(args.mapping, profile["mapping"])
    print(f"pad profile: {profile['label']}  [{why}]")
    scale = min(1.0, max(0.05, args.scale))

    try:
        arm = ArmConnection(args.robot_ip, args.username, args.password,
                            dry_run=args.dry_run)
    except Exception as exc:                          # noqa: BLE001
        pad.close()
        print(f"\nerror: could not connect to the arm at {args.robot_ip}: {exc}",
              file=sys.stderr)
        print("       check the static IP on the wired interface and that",
              file=sys.stderr)
        print("       'ping 192.168.1.10' replies; see the docstring.",
              file=sys.stderr)
        return 3

    if args.dry_run:
        print("DRY RUN -- no connection, no motion. Printing commands only.")
    else:
        print(f"connected to {args.robot_ip}, {arm.actuator_count} actuators")
        arm.report_faults()

    frame_base = not args.tool_frame
    print(f"cartesian frame: {'BASE' if frame_base else 'TOOL'}")
    print(f"speed scale: {scale:.0%}   "
          f"(max {args.max_lin} m/s, {args.max_ang} deg/s, "
          f"{args.max_joint} deg/s per joint)")
    if args.no_deadman:
        print("!! --no-deadman: motion does NOT require holding RB !!")
    else:
        print(f"HOLD button {mapping['deadman']} (RB) DOWN while steering -- "
              f"sticks and triggers do nothing without it.")
        print("BACK to quit.  A = stop, Y = clear faults.")
    print("status flags:  [---] deadman released (sticks ignored)   "
          "[rdy] deadman held   [GO ] commanding")

    joint_mode = False
    sel_joint = 0
    dpad_x_prev = 0.0
    dpad_y_prev = 0.0
    period = 1.0 / args.rate
    last_status = 0.0
    last_hint = 0.0
    was_active = False
    rc = 0

    try:
        while True:
            loop_start = time.time()

            if not pad.poll():
                print("\ngamepad disconnected -- stopping arm")
                break

            # Every pressed() must be evaluated each frame or its edge state
            # goes stale, so read them all before acting on any.
            quit_pressed = pad.pressed(mapping["quit"])
            stop_pressed = pad.pressed(mapping["stop"])
            if quit_pressed:
                print("\nBACK pressed -- quitting")
                break
            if stop_pressed:
                arm.stop()
                print("\n[A] hard stop")
            if pad.pressed(mapping["clear_faults"]):
                arm.clear_faults()
            if pad.pressed(mapping["toggle_joint"]):
                joint_mode = not joint_mode
                arm.stop()
                print(f"\nmode: {'JOINT' if joint_mode else 'CARTESIAN'}")

            # D-pad: up/down = speed scale, left/right = joint select. The
            # F710 reports the D-pad as two axes that snap to +/-1, so treat a
            # sign change as a discrete press.
            dpad_y = -pad.axis(mapping["dpad_y"])   # inverted: up = +1
            dpad_x = pad.axis(mapping["dpad_x"])
            if dpad_y > 0.5 >= dpad_y_prev:
                scale = min(1.0, scale + 0.1)
                print(f"\nspeed scale: {scale:.0%}")
            elif dpad_y < -0.5 <= dpad_y_prev:
                scale = max(0.05, scale - 0.1)
                print(f"\nspeed scale: {scale:.0%}")
            if joint_mode:
                if dpad_x > 0.5 >= dpad_x_prev:
                    sel_joint = (sel_joint + 1) % arm.actuator_count
                    print(f"\nselected joint {sel_joint + 1}")
                elif dpad_x < -0.5 <= dpad_x_prev:
                    sel_joint = (sel_joint - 1) % arm.actuator_count
                    print(f"\nselected joint {sel_joint + 1}")
            dpad_y_prev, dpad_x_prev = dpad_y, dpad_x

            deadman = args.no_deadman or pad.button(mapping["deadman"])
            rotate = pad.button(mapping["rotate_mod"])

            def stick(name):
                raw = pad.axis(mapping[name])
                if name in INVERTED_AXES:
                    raw = -raw
                return apply_deadzone(raw)

            lin = (0.0, 0.0, 0.0)
            ang = (0.0, 0.0, 0.0)
            joints = [0.0] * arm.actuator_count
            grip = 0.0

            if deadman:
                # XInput triggers are analog axes resting at -1; DirectInput
                # has no analog triggers at all and reports LT/RT as buttons.
                # Both pressed cancels rather than arbitrarily picking one.
                if analog_triggers:
                    close = pad.axis(mapping["trigger_left"]) > TRIGGER_ON
                    open_ = pad.axis(mapping["trigger_right"]) > TRIGGER_ON
                else:
                    close = pad.button(mapping["trigger_left"])
                    open_ = pad.button(mapping["trigger_right"])
                if close and not open_:
                    grip = GRIPPER_SPEED
                elif open_ and not close:
                    grip = -GRIPPER_SPEED

                if joint_mode:
                    joints[sel_joint] = stick("lin_x") * args.max_joint * scale
                elif rotate:
                    ang = (stick("lin_x") * args.max_ang * scale,
                           stick("lin_y") * args.max_ang * scale,
                           stick("ang_z") * args.max_ang * scale)
                else:
                    lin = (stick("lin_x") * args.max_lin * scale,
                           stick("lin_y") * args.max_lin * scale,
                           stick("lin_z") * args.max_lin * scale)

            # Sticks and triggers are gated on the deadman, the D-pad and face
            # buttons are not. That asymmetry reads as "the sticks are broken"
            # if the deadman is not actually held, so say what is happening
            # rather than silently commanding zero.
            if not deadman:
                deflected = (abs(stick("lin_x")) > 0 or abs(stick("lin_y")) > 0
                             or abs(stick("lin_z")) > 0 or abs(stick("ang_z")) > 0)
                if deflected and time.time() - last_hint > 3.0:
                    last_hint = time.time()
                    print(f"\n[hint] stick moved with the deadman released -- "
                          f"nothing will move. HOLD button {mapping['deadman']} "
                          f"(RB) DOWN while you steer.")

            active = deadman and (any(lin) or any(ang) or any(joints) or grip)

            if joint_mode:
                arm.send_joint_speeds(joints)
            else:
                arm.send_twist(lin, ang, frame_base=frame_base)
            # Only touch the gripper when the command changes or is nonzero --
            # a 40 Hz stream of zero-speed gripper commands is pointless
            # traffic and makes the finger buzz on some firmware.
            if grip or was_active:
                arm.send_gripper_speed(grip)
            was_active = bool(grip)

            now = time.time()
            if now - last_status > 0.1:
                last_status = now
                if joint_mode:
                    body = (f"JOINT j{sel_joint + 1} "
                            f"{joints[sel_joint]:+6.1f} deg/s")
                else:
                    body = (f"CART lin {lin[0]:+.3f} {lin[1]:+.3f} {lin[2]:+.3f} m/s"
                            f"  ang {ang[0]:+5.1f} {ang[1]:+5.1f} {ang[2]:+5.1f} d/s")
                flag = "GO " if active else ("rdy" if deadman else "---")
                sys.stdout.write(f"\r[{flag}] {scale:>4.0%} {body}  grip {grip:+.1f}  ")
                sys.stdout.flush()

            sleep = period - (time.time() - loop_start)
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\nCtrl-C -- stopping arm")
    except Exception as exc:                          # noqa: BLE001
        print(f"\nerror in teleop loop: {exc}")
        rc = 1
    finally:
        # Always try to stop, even if the loop died mid-motion.
        try:
            arm.stop()
        finally:
            arm.close()
            pad.close()
        print("\nstopped, session closed")

    return rc


if __name__ == "__main__":
    sys.exit(main())
