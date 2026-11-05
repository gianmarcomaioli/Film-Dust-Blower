# Film Dust Blower by @GianmarcoMaioli

A Lightroom Classic plugin that finds dust specks, hairs and faint soft dust on
film scans and removes them with Lightroom's own heal spots. The repairs are
exactly what you would get with the Spot Removal tool:
visible in Develop, adjustable one by one, and undoable in History.

## Requirements
- Lightroom Classic
- Python 3.8 or later with NumPy, OpenCV and tifffile:
      py -m pip install numpy opencv-python tifffile        (Windows)
      python3 -m pip install numpy opencv-python tifffile   (Mac)

## Installing

1. Unzip, and put the folder that directly contains `Info.lua` somewhere permanent.
2. In Lightroom: File > Plug-in Manager > Add, and choose that folder.
3. Open File > Plug-in Extras > Film Dust Blower: Settings... and click **Test**.

## Using it
All commands are under **File > Plug-in Extras** and work in any module.

- **Preview Dust on Selected...** changes nothing. It saves a dust map per photo, 
with red circles for specks and blue boxes for hairs, and opens the folder.
- **Clean Selected Photos** adds heal spots to each selected photo. 
Your already existing spots are kept.

## The Process
For each photo, cleaning works in four stages:

1. It briefly switches off the crop, lens corrections and transforms
(history step "Film Dust Blower: prepare").
2. It renders the photo at full resolution and finds the dust inside your crop.
3. It puts your crop, lens corrections and transforms back.
4. It adds the heal spots, all in one step ("Film Dust Blower").

**To undo**, click the history step just before "Film Dust Blower: prepare",
or press Ctrl+Z | Cmd+Z twice. You can also select and delete individual spots with
the Spot Removal tool.

I recommend using it after the image has been converted, but it can be done before too.

## Settings

| Setting | What it does |
|---------|--------------|
| Repair as | Heal spots on the photo (default), or a separate cleaned 16-bit TIFF stacked above the original. |
| Dust shows | Automatic, Dark (raw negatives, slides) or Light (converted negatives). |
| Sensitivity | 1.0 is balanced. Raise it to also catch fainter dust; lower it to heal only the most obvious. |
| Largest speck | Bigger marks are left alone, as a percentage of the frame's short side. |
| Remove hairs and fibres | Hairs become brush strokes along their length. |
| Most spots per photo | Hundreds of spots slow Lightroom's Develop module down. |
| Save dust maps | When cleaning, also keep the maps of what was found. |

## How it works

1. **Candidates.** At several scales, a morphological filter finds every mark
that is smaller and lighter, or darker, than its surroundings. A small
crisp speck inside a larger soft patch is judged on its own.
2. **Measurements.** Each candidate is measured about thirty ways:
- its contrast against the grain, size, isolation and colour;
- the structure around it (edges, branches, windows);
- **how sharp it is, and whether it is sharper than everything around it at
its own size. ** Dust lies on the film, so it is as sharp as the grain;
what the lens drew is softer. 
3. **The decision.** A small decision-tree model (`dust_model.json`) trained on
frames cleaned by hand decides, with three rules on top:
- only clearly visible marks are healed (the kind you would heal yourself);
- where the surroundings are strongly structured, only marks the model is very sure of;
- a small, clearly visible light mark that is much sharper than everything
around it is healed even if the model hesitates (unless the surroundings are structured).
4. **Soft discs.** A separate search at quarter size, where the grain averages
out, finds faint, round, soft lifts on smooth ground: out-of-focus dust, and
soft patches like the ones hard water leaves in skies.
5. **Spots.** Each accepted speck or disc becomes one heal spot, and each hair a
brush stroke. The source is the nearby patch whose surroundings match best.

## Limitations

- Faint soft patches are found when they are round and on smooth ground. Larger
 or irregular water marks are not yet.
- In busy, sharp areas (foliage in focus, gravel) small dust is deliberately
left alone, because there it looks just like a lit leaf.
- Large scratches and long straight lines are not treated as dust.
- The heal-spot format is the one Lightroom 14 uses. After cleaning, the plugin
checks that Lightroom kept the spots and tells you if it didn't. In that case,
 use "Repair as: Separate cleaned TIFF copy".

## Credits

Two ideas come from **film dust**, an open-source Photoshop plugin by Magnus
Holmes (https://github.com/magholmes/film-dust, MIT licence, copyright (c) 2026
Magnus Holmes): measuring a mark's crispness as the share of its contrast that
survives a fine high-pass, compared with a perfectly sharp mark of the same
shape; and searching for faint soft discs at reduced size, with a difference of
Gaussians per scale measured against that scale's own noise. Both are
re-implemented here in my own code, without using any film dust code.

Film Dust Blower was created by Gianmarco Maioli
find more at Youtube @gianmarcomaioli or Instagram @gianmarcomaioli

Copyright (c) 2026 Gianmarco Maioli
released under GPL-3.0 licence
Film Dust Blower is free and open-source software. 
You may use, copy, modify and share it, provided the implementation
of its code remains free and sourceable.