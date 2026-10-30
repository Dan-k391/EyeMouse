"""macOS cursor, click and hotkey access through Quartz. Moving the cursor needs Accessibility permission."""
from AppKit import NSSound
import Quartz as Q

ESCAPE_KEY = 53
_MODIFIERS = dict(control=Q.kCGEventFlagMaskControl, option=Q.kCGEventFlagMaskAlternate,
                  command=Q.kCGEventFlagMaskCommand)


def has_access():
    return bool(Q.CGPreflightPostEventAccess())


def request_access():
    Q.CGRequestPostEventAccess()


def screen_size():
    bounds = Q.CGDisplayBounds(Q.CGMainDisplayID())
    return int(bounds.size.width), int(bounds.size.height)


def position():
    point = Q.CGEventGetLocation(Q.CGEventCreate(None))
    return point.x, point.y


def _post(kind, x, y, button=Q.kCGMouseButtonLeft):
    Q.CGEventPost(Q.kCGHIDEventTap, Q.CGEventCreateMouseEvent(None, kind, (float(x), float(y)), button))


def move(x, y):
    _post(Q.kCGEventMouseMoved, x, y)


def click(x, y, right=False):
    down, up, button = ((Q.kCGEventRightMouseDown, Q.kCGEventRightMouseUp, Q.kCGMouseButtonRight) if right
                        else (Q.kCGEventLeftMouseDown, Q.kCGEventLeftMouseUp, Q.kCGMouseButtonLeft))
    _post(down, x, y, button)
    _post(up, x, y, button)


def modifiers():
    """Held modifier keys. Polling state works in any app and needs no event tap."""
    flags = Q.CGEventSourceFlagsState(Q.kCGEventSourceStateHIDSystemState)
    return {name for name, mask in _MODIFIERS.items() if flags & mask}


def key_down(code):
    return bool(Q.CGEventSourceKeyState(Q.kCGEventSourceStateHIDSystemState, code))


def sound(name):
    player = NSSound.soundNamed_(name)
    if player:
        player.stop()
        player.play()
