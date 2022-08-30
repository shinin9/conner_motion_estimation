from time import sleep
from bhaptics import better_haptic_player as player
import keyboard

player.initialize()
sleep(10)

def play_haptic(idx_list, vestIdx):
    dotFrame_F = {
        "Position": "VestFront",
        "DotPoints": [],
        "DurationMillis": 1000
    }
    dotFrame_B = {
        "Position": "VestBack",
        "DotPoints": [],
        "DurationMillis": 1000
    }

    for i in idx_list:
        if i in [2, 3, 4, 5, 10, 11, 12, 13, 18, 19, 20, 21, 26, 27, 28, 29, 34, 35, 36, 37]:
            dotPint = {
                "Index": int(2),
                "Intensity": int(vestIdx[i])
            }
            dotFrame_F["DotPoints"].append(dotPint)
        else:
            dotPint = {
                "Index": int(2),
                "Intensity": int(vestIdx[i])
            }
            dotFrame_B["DotPoints"].append(dotPint)
    # for i in idx_list:
    #     dotPint = {
    #         "Index": int(i),
    #         "Intensity": int(vestIdx[i])
    #     }
    #     dotFrame_F["DotPoints"].append(dotPint)

    player.submit("dotPoint_F", dotFrame_F)
    player.submit("dotPoint_B", dotFrame_B)
    # sleep(0.1)







