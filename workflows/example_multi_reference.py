#!/usr/bin/env python3
"""Old Man Rivers — full six-reference build.

Every character, vehicle and location Tony supplied gets its own reference so
nothing is invented and nothing drifts between cuts. That is the whole reason
for the extra refs: un-referenced subjects are exactly where these models wander.

Usage: rivers_full.py <host:port> [w] [h] [len]
"""
import json
import sys
import urllib.request

HOST = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:8188"
W = int(sys.argv[2]) if len(sys.argv) > 2 else 832
H = int(sys.argv[3]) if len(sys.argv) > 3 else 480
L = int(sys.argv[4]) if len(sys.argv) > 4 else 362      # 17*21+5 = 15.08s @24fps

REFS = ["rivers_ref.jpg", "street_ref.jpg", "house_ref.jpg",
        "car_ref.jpg", "mom_ref.jpg", "johnny_ref.jpg"]

PROMPT = """integrated_multimodal_description: 8K IMAX. Photorealistic — no 3D render, no game engine, no game-cutscene aesthetic. Cinematography: Roger Deakins x Gregory Crewdson. Strictly practical light sources physically present on set — flat overcast afternoon daylight, one weak yellow bulb burning on the porch in daytime, zero added cinematic fills, no front fill, no rim light from outside frame. Color 60:30:10, sun-bleached early-2000s palette, warm asphalt, dry lawns, washed sky. Physical cine lens, 180° shutter motion blur. Realism: pore-level skin, vellus hair, liver spots, dust and pollen in the air, real glass with real smears and real reflections, gravity and inertia fully respected, correct contact shadows. Continuity identical across cuts, no identity drift — same faces, same clothes, same car, same houses in every shot. 24fps. 8K detail.

Reference picture 1 is Old Man Rivers, three angles. Very old, gaunt, tall but hunched. Sparse long grey hair. Deeply weathered sun-damaged skin, liver spots and scabs across his shoulders and arms. Both eyes completely clouded milk-white and blind. Mouth hanging open around a few long protruding yellow-brown teeth, lower lip slack and wet. Shirtless under filthy dark denim bib overalls with one strap unfastened and hanging, knees torn out. Bare dirty feet.

Reference picture 2 is the street — a quiet early-2000s residential road, neat two-storey vinyl-sided houses, mown lawns, mailboxes on posts, mature trees, cracked sidewalk, parked sedans in driveways.

Reference picture 3 is Rivers' house — a derelict weathered timber shack with a sagging covered porch, split shingle roof, boarded and broken windows, a crumbling brick chimney, standing alone on bare dirt and dead grass. It is the only structure on the street that looks like this.

Reference picture 4 is the family car — a gold-beige late-1990s four-door sedan, dusty.

Reference picture 5 is the mother, three angles. Mid-forties, shoulder-length auburn hair, dark maroon cardigan over a cream top, blue jeans, brown leather boots.

Reference picture 6 is Johnny, three angles. A teenage boy, roughly fifteen, light brown hair, dark green zip hoodie over a grey t-shirt, loose blue jeans, white sneakers.

[Shot 1] A moving exterior shot tracking alongside the gold-beige sedan from reference picture 4 as it rolls at walking pace down the street from reference picture 2, camera level with the car, shooting THROUGH the passenger glass — real reflections of houses and telephone poles sliding across the window over the people inside. The mother from reference picture 5 is driving, cheerful, glancing at the rear-view mirror. Johnny from reference picture 6 sits in the back seat looking out of his window, headphones around his neck. Warm, ordinary, unremarkable. She says brightly, half over her shoulder: <d>[English] You're going to love it here, Johnny. Wait till you see the yard.</d> He answers flatly, not looking at her: <d>[English] I don't know, Mom.</d>

[Shot 2] At 00:04.000, hard cut to a tight shot on Johnny through the same passing glass, reflections still crawling over his face. He turns his head and looks up and out of the window at something off to the side of the car. His expression empties. His mother's talk continues as muffled unintelligible sound in the front seat. He does not blink, does not speak, and the car keeps rolling, carrying him slowly past whatever he is looking at.

[Shot 3] At 00:07.000, hard cut to Johnny's eyeline — a slow push-in across a dead lawn toward the derelict shack from reference picture 3, sitting wrong among the neat houses on either side of it. Old Man Rivers from reference picture 1 stands hunched far over the sagging porch railing, upper body tipped forward at a wrong angle, arms hanging, both blind milk-white eyes aimed directly at the camera, mouth slack and open around the long yellow teeth, a thread of drool catching the daylight. He is completely motionless except for his chest, which works hard and fast and shallow. He tracks the car without moving his head. The camera holds on him as the yellow porch bulb buzzes above his shoulder, and it does not cut away.

Camera: three fixed setups, hard cuts between them, no dissolves. Shots 1 and 2 move with the car; shot 3 is a slow steady push-in that never retreats.

overall_soundscape: Environment SFX only, no music, no subtitles. Tyres crawling over warm asphalt, the sedan's tired engine, a turn signal ticking, cicadas in the heat, a dog barking two streets over, a wind chime. Inside the car, muffled radio and the mother's voice going indistinct after her line. Across the lawn, Old Man Rivers' breathing — a fast, wet, rattling open-mouthed pant, ragged and effortful, mixed unnaturally close and dry as if he were standing beside the microphone rather than across the yard, rising in the last seconds. The porch bulb buzzes. No footsteps, no rustle, no movement sound from him at all.

non_diegetic_music: N/A"""


def build(seed=7731, prefix="video/rivers_porch_480"):
    g = {
        "6": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                         "weight_dtype": "default"}},
        "13": {"class_type": "CLIPLoader",
               "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                          "type": "minimax", "device": "default"}},
        "11": {"class_type": "VAELoader",
               "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "24": {"class_type": "VAELoader",
               "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
    }
    # One LoadImage node per reference; ref_images takes the list of NODE ids.
    ref_nodes = []
    for i, name in enumerate(REFS):
        nid = str(201 + i)
        g[nid] = {"class_type": "LoadImage",
                  "inputs": {"image": name, "upload": "image"}}
        ref_nodes.append(nid)
    g["230"] = {"class_type": "MiniMaxH3ReferenceToVideo",
                "inputs": {"clip": ["13", 0], "vae": ["11", 0], "audio_vae": ["24", 0],
                           "ref_images": ref_nodes, "prompt": PROMPT,
                           "width": W, "height": H, "length": L,
                           "ref_image_size": "max"}}
    g.update({
        "16": {"class_type": "BasicGuider",
               "inputs": {"model": ["6", 0], "conditioning": ["230", 0]}},
        "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "9": {"class_type": "BasicScheduler",
              "inputs": {"model": ["6", 0], "scheduler": "simple",
                         "steps": 20, "denoise": 1.0}},
        "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "14": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["15", 0], "guider": ["16", 0],
                          "sampler": ["17", 0], "sigmas": ["9", 0],
                          "latent_image": ["230", 1]}},
        "10": {"class_type": "VAEDecode",
               "inputs": {"samples": ["14", 0], "vae": ["11", 0]}},
        # Audio decodes off the SAME sampler latent as the video, not off a
        # third output of 230 — that node returns only CONDITIONING and LATENT.
        "23": {"class_type": "VAEDecodeAudio",
               "inputs": {"samples": ["14", 0], "vae": ["24", 0]}},
        "91": {"class_type": "CreateVideo",
               "inputs": {"images": ["10", 0], "audio": ["23", 0], "fps": 24.0}},
        "92": {"class_type": "SaveVideo",
               "inputs": {"video": ["91", 0], "filename_prefix": prefix,
                          "format": "auto", "codec": "auto"}},
    })
    return g


if __name__ == "__main__":
    body = json.dumps({"prompt": build(), "client_id": "kai-rivers"}).encode()
    req = urllib.request.Request(f"http://{HOST}/prompt", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            print(json.dumps(json.loads(r.read().decode()), indent=1))
    except urllib.error.HTTPError as e:
        print("HTTP", e.code)
        print(e.read().decode()[:2500])
