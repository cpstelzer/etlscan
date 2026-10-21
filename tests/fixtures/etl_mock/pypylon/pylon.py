"""Fake pypylon.pylon with realistic free-run / trigger timing (see sim_hw)."""
import threading
import time
from collections import deque

import sim_hw
from sim_hw import LENS, render

pc = time.perf_counter

GrabStrategy_OneByOne = 0
GrabStrategy_LatestImageOnly = 1
GrabStrategy_LatestImages = 2
GrabStrategy_UpcomingImage = 3
TimeoutHandling_ThrowException = 0
TimeoutHandling_Return = 1


class GenericException(Exception):
    pass


class TimeoutException(GenericException):
    pass


class Node:
    def __init__(self, value=None, mn=None, mx=None, inc=1, symbolics=None):
        self.Value = value
        self.Min, self.Max, self.Inc = mn, mx, inc
        self.HasMax = mx is not None
        self.Symbolics = symbolics or []

    def GetValue(self):
        return self.Value

    def SetValue(self, v):
        self.Value = v

    def IsReadable(self):
        return True


class DynNode(Node):
    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def GetValue(self):
        return self._fn()


class Cmd:
    def __init__(self, fn):
        self.fn = fn

    def Execute(self):
        self.fn()


class GrabResult:
    def __init__(self, block_id, ticks, image, ok=True, valid=True):
        self._b, self._t, self.Array, self._ok, self._valid = block_id, ticks, image, ok, valid

    def GrabSucceeded(self):
        return self._ok

    def IsValid(self):
        return self._valid

    def IsReadable(self):
        return self._valid

    def GetBlockID(self):
        return self._b

    @property
    def TimeStamp(self):
        return self._t

    def Release(self):
        pass

    def GetErrorCode(self):
        return 0

    def GetErrorDescription(self):
        return ""


class _Info:
    def GetModelName(self):
        return "SimCam a2A4504-27g5m"

    def GetSerialNumber(self):
        return "0000"


class InstantCamera:
    def __init__(self, dev=None):
        self._open = False
        self.PixelFormat = Node("Mono8", symbolics=["Mono8", "Mono12"])
        self.ExposureTime = Node(750.0, 20.0, 1e6)
        self.WidthMax, self.HeightMax = Node(4512), Node(4512)
        self.Width, self.Height = Node(4512, 16, 4512, 4), Node(4512, 16, 4512, 4)
        self.OffsetX, self.OffsetY = Node(0, 0, 4512, 4), Node(0, 0, 4512, 4)
        self.AcquisitionFrameRateEnable = Node(True)
        self.AcquisitionFrameRate = Node(25.0, 0.1, 27.0)
        self.ResultingFrameRate = Node(25.0)
        self.TriggerSelector = Node("FrameStart")
        self.TriggerMode = Node("Off")
        self.TriggerSource = Node("Software")
        self.MaxNumBuffer = Node(50)
        self.NumReadyBuffers = DynNode(self._ready_count)
        self.TimestampLatchValue = Node(0)
        self.TimestampLatch = Cmd(self._latch)
        self._q = deque()
        self._lock = threading.Lock()
        self._grabbing = False
        self._thread = None
        self._block = 0
        self._last_trigger = -1e9

    def __setattr__(self, name, value):
        # pypylon: assigning a plain value to a node attribute sets the node (camera.MaxNumBuffer = 50)
        node = self.__dict__.get(name)
        if isinstance(node, Node) and not isinstance(value, Node):
            node.SetValue(value)
        else:
            object.__setattr__(self, name, value)

    def _latch(self):
        self.TimestampLatchValue.Value = int(pc() * 1e9) + sim_hw.CLOCK_OFFSET_TICKS

    def GetDeviceInfo(self):
        return _Info()

    def Open(self):
        self._open = True

    def Close(self):
        self._open = False

    def IsOpen(self):
        return self._open

    def _expose(self, t):
        img = render(LENS.actual_at(t))
        self._block += 1
        if self._block in sim_hw.DROP_BLOCK_IDS:
            return
        with self._lock:
            self._q.append((t + sim_hw.TRANSFER_S, GrabResult(self._block, int(t * 1e9) + sim_hw.CLOCK_OFFSET_TICKS, img)))

    def _freerun(self):
        period = 1.0 / float(self.AcquisitionFrameRate.Value)
        t = pc() + 0.005
        while self._grabbing:
            now = pc()
            if now >= t:
                self._expose(t)
                t += period
            time.sleep(0.0005)

    def StartGrabbing(self, strategy=0):
        self._grabbing = True
        self._block = 0
        self._q.clear()
        if self.TriggerMode.Value == "Off":
            self._thread = threading.Thread(target=self._freerun, daemon=True)
            self._thread.start()

    def StopGrabbing(self):
        self._grabbing = False
        if self._thread:
            self._thread.join()
        self._thread = None

    def IsGrabbing(self):
        return self._grabbing

    def _ready_count(self):
        now = pc()
        with self._lock:
            return sum(1 for ta, _ in self._q if ta <= now)

    def RetrieveResult(self, timeout_ms, handling=0):
        end = pc() + timeout_ms / 1000.0
        while pc() < end:
            with self._lock:
                if self._q and self._q[0][0] <= pc():
                    return self._q.popleft()[1]
            time.sleep(0.0003)
        if handling == TimeoutHandling_Return:
            return GrabResult(None, None, None, ok=False, valid=False)
        raise TimeoutException("timeout")

    def _ready_count_unlocked(self):
        now = pc()
        return sum(1 for ta, _ in self._q if ta <= now)

    def WaitForFrameTriggerReady(self, timeout_ms, handling=0):
        while pc() < self._last_trigger + sim_hw.MIN_PERIOD_S:
            time.sleep(0.0002)
        return True

    def ExecuteSoftwareTrigger(self):
        t = pc()
        self._last_trigger = t
        self._expose(t)


class _Factory:
    def EnumerateDevices(self):
        return ["sim"]

    def CreateDevice(self, d):
        return d


class TlFactory:
    @staticmethod
    def GetInstance():
        return _Factory()
