from gymnasium import register
import gymnasium as gym
from minigrid.utils.window import Window
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

register(
    'MOLTLTest-6x6-v0', 
    entry_point='mappo.tests.multi_objective:MOLTLTestEnv'
)



def redraw(window, img):
    window.show_img(img)


def reset(env, window, seed=None):
    env.reset()

    if hasattr(env, "mission"):
        print("Mission: %s" % env.mission)
        window.set_caption(env.mission)

    img = env.get_frame()

    redraw(window, img)


def step(env, window, action):
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"step={env.step_count}, reward={reward:.2f}")

    if terminated:
        print("terminated!")
        reset(env, window)
    elif truncated:
        print("truncated!")
        reset(env, window)
    else:
        img = env.get_frame()
        redraw(window, img)


def key_handler(env, window, event):
    print("pressed", event.key)

    if event.key == "escape":
        window.close()
        return

    if event.key == "backspace":
        reset(env, window)
        return

    if event.key == "left":
        step(env, window, env.actions.left)
        return
    if event.key == "right":
        step(env, window, env.actions.right)
        return
    if event.key == "up":
        step(env, window, env.actions.forward)
        return

    # Spacebar
    if event.key == " ":
        step(env, window, env.actions.toggle)
        return
    if event.key == "pageup":
        step(env, window, env.actions.pickup)
        return
    if event.key == "pagedown":
        step(env, window, env.actions.drop)
        return

    if event.key == "enter":
        step(env, window, env.actions.done)
        return


if __name__ == "__main__":
    import argparse

    env = gym.make("MOLTLTest-6x6-v0")
    
