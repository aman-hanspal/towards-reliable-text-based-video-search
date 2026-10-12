# Sanity sample (v3): 30 random videos

For each HARD-absent prompt ask: *could this prompt honestly describe something that is in the video, or be its parent/child?* If yes -> UNFAIR. A fair prompt for an object that might be present but unlabelled is NOT unfair; that is caught later at verify-on-fire.

## LVVIS/train_00265  (video_id 245, 15 frames)
- present: beachball [base], boot [base], jacket [base], person [base], shoe [novel], tights [novel], trousers [novel]
- easy absent: computer keyboard, cod
- hard absent: box (anchor 'boot', wordnet_cousin, wup 0.857); sportswear (anchor 'tights', wordnet_cousin, wup 0.762)
- your verdict: fair / unfair because ...

## LVVIS/val_00171  (video_id 1578, 15 frames)
- present: folding knife [novel], person [base]
- easy absent: board eraser, rice cooker
- hard absent: shears (anchor 'folding knife', wordnet_cousin, wup 0.917); sword (anchor 'folding knife', wordnet_cousin, wup 0.9)
- your verdict: fair / unfair because ...

## LVVIS/train_00487  (video_id 338, 20 frames)
- present: beluga whale [novel]
- easy absent: kiwi fruit, infusion pump
- hard absent: seal (anchor 'beluga whale', wordnet_cousin, wup 0.857); polar bear (anchor 'beluga whale', wordnet_cousin, wup 0.786)
- your verdict: fair / unfair because ...

## LVVIS/train_02383  (video_id 1196, 21 frames)
- present: box [base], dehumidifier [novel], desk [base], person [base], speaker [novel], thermometer [novel]
- easy absent: headboard, tarp
- hard absent: clock (anchor 'thermometer', wordnet_cousin, wup 0.857); cabinet (anchor 'desk', wordnet_cousin, wup 0.842)
- your verdict: fair / unfair because ...

## LVVIS/val_00032  (video_id 1510, 15 frames)
- present: burette [novel], person [base]
- easy absent: mashed potato, couch
- hard absent: sword (anchor 'burette', wordnet_cousin, wup 0.8); igniter (anchor 'burette', wordnet_cousin, wup 0.778)
- your verdict: fair / unfair because ...

## LVVIS/train_00653  (video_id 410, 30 frames)
- present: gun [base], person [base], pistol [base], rifle [novel], watch [novel]
- easy absent: firefly, fig
- hard absent: alarm clock (anchor 'watch', wordnet_cousin, wup 0.87); knife (anchor 'pistol', wordnet_cousin, wup 0.818)
- your verdict: fair / unfair because ...

## LVVIS/train_00225  (video_id 225, 10 frames)
- present: person [base], swim cap [novel], swimming goggles [novel], swimsuit [novel]
- easy absent: tape measure, paper bag
- hard absent: helmet (anchor 'swim cap', wordnet_cousin, wup 0.9); dress hat (anchor 'swim cap', wordnet_cousin, wup 0.857)
- your verdict: fair / unfair because ...

## LVVIS/train_00566  (video_id 367, 24 frames)
- present: bible [base], book [base], computer keyboard [base], monitor [novel], person [base], ring [novel], table [novel]
- easy absent: electric heater, halter top
- hard absent: joystick (anchor 'computer keyboard', wordnet_cousin, wup 0.909); bracelet (anchor 'ring', wordnet_cousin, wup 0.889)
- your verdict: fair / unfair because ...

## LVVIS/train_02741  (video_id 1344, 25 frames)
- present: coatrack [base], dress [base], lab coat [base], person [base], spectacles [novel]
- easy absent: polo shirt, ashtray
- hard absent: jersey (anchor 'dress', wordnet_cousin, wup 0.8); shirt (anchor 'dress', wordnet_cousin, wup 0.842)
- your verdict: fair / unfair because ...

## UVO/-6k2ntyDP-c  (video_id 1922, 90 frames)
- present: frisbee [base], person [base], potted plant [novel], tv [novel]
- easy absent: mixer, kimono
- hard absent: camera (anchor 'tv', wordnet_cousin, wup 0.762); bowl (anchor 'potted plant', wordnet_cousin, wup 0.8)
- your verdict: fair / unfair because ...

## LVVIS/val_00259  (video_id 1624, 24 frames)
- present: handcuff [base], key [base], person [base]
- easy absent: hand towel, microscope
- hard absent: clasp (anchor 'handcuff', wordnet_cousin, wup 0.8); toy (anchor 'key', wordnet_cousin, wup 0.875)
- your verdict: fair / unfair because ...

## LVVIS/train_02869  (video_id 1399, 30 frames)
- present: box [base], person [base], remote control car [novel]
- easy absent: correction fluid, wood plane
- hard absent: motorcycle (anchor 'remote control car', wordnet_cousin, wup 0.917); can (anchor 'box', wordnet_cousin, wup 0.875)
- your verdict: fair / unfair because ...

## LVVIS/train_00053  (video_id 143, 25 frames)
- present: loader [novel], person [base], truck [novel], wheel [novel]
- easy absent: knee pad, fig
- hard absent: ambulance (anchor 'truck', wordnet_cousin, wup 0.88); drum (anchor 'wheel', wordnet_cousin, wup 0.778)
- your verdict: fair / unfair because ...

## AnimalTrack/dolphin_4  (video_id 8, 208 frames)
- present: dolphin [base]
- easy absent: potholder, eraser
- hard absent: elephant (anchor 'dolphin', wordnet_cousin, wup 0.759); salmon (anchor 'dolphin', wordnet_cousin, wup 0.812)
- your verdict: fair / unfair because ...

## LVVIS/train_01737  (video_id 904, 17 frames)
- present: crib [base], person [base]
- easy absent: bamboo, sphygmomanometer
- hard absent: crate (anchor 'crib', wordnet_cousin, wup 0.778); box (anchor 'crib', wordnet_cousin, wup 0.824)
- your verdict: fair / unfair because ...

## LVVIS/train_00348  (video_id 282, 24 frames)
- present: banjo [base], chair [base], cushion [base], doorknob [base], green plants [novel], mug [base], person [base], skirt [novel], table [novel]
- easy absent: carabiner, squid
- hard absent: bowl (anchor 'mug', wordnet_cousin, wup 0.842); footstool (anchor 'table', wordnet_cousin, wup 0.8)
- your verdict: fair / unfair because ...

## LVVIS/train_02192  (video_id 1121, 18 frames)
- present: car [novel], hot-air balloon [base], minivan [base], telephone pole [novel]
- easy absent: harmonium, spatula
- hard absent: lamppost (anchor 'telephone pole', wordnet_cousin, wup 0.917); helicopter (anchor 'hot-air balloon', wordnet_cousin, wup 0.8)
- your verdict: fair / unfair because ...

## LVVIS/train_01336  (video_id 721, 30 frames)
- present: glove [base], lasagna [base], person [base], plate [base], watch [novel]
- easy absent: eye liner, mechanical pencil
- hard absent: alarm clock (anchor 'watch', wordnet_cousin, wup 0.87); clock (anchor 'watch', wordnet_cousin, wup 0.909)
- your verdict: fair / unfair because ...

## LVVIS/val_00734  (video_id 1854, 15 frames)
- present: bowl [base], plate [base], soya milk [novel], spoon [novel]
- easy absent: zebra, lighthouse
- hard absent: fork (anchor 'spoon', wordnet_cousin, wup 0.9); tray (anchor 'plate', wordnet_cousin, wup 0.889)
- your verdict: fair / unfair because ...

## LVVIS/train_01214  (video_id 661, 16 frames)
- present: bottle [base], flower arrangement [base], person [base], shampoo [novel], shower cap [novel], shower head [novel], towel rack [novel]
- easy absent: seashell, hippopotamus
- hard absent: coatrack (anchor 'towel rack', wordnet_cousin, wup 0.9); cup (anchor 'bottle', wordnet_cousin, wup 0.824)
- your verdict: fair / unfair because ...

## LVVIS/train_01824  (video_id 945, 25 frames)
- present: excavator [novel], truck [novel]
- easy absent: hand grips strengthener, oxygen concentrator
- hard absent: car (anchor 'truck', wordnet_cousin, wup 0.917); power saw (anchor 'excavator', wordnet_cousin, wup 0.842)
- your verdict: fair / unfair because ...

## LVVIS/val_00759  (video_id 1865, 35 frames)
- present: hotplate [base], knob [base]
- easy absent: tassel, swing
- hard absent: ring (anchor 'knob', wordnet_cousin, wup 0.75); gas stove (anchor 'hotplate', wordnet_cousin, wup 0.88)
- your verdict: fair / unfair because ...

## LVVIS/train_00813  (video_id 481, 24 frames)
- present: banjo [base], earring [base], person [base], ring [novel], strap [novel]
- easy absent: platypus, cheetah
- hard absent: guitar (anchor 'banjo', wordnet_cousin, wup 0.9); pin (anchor 'ring', wordnet_cousin, wup 0.889)
- your verdict: fair / unfair because ...

## LVVIS/train_01427  (video_id 764, 12 frames)
- present: steamroller [novel]
- easy absent: octopus, bathrobe
- hard absent: bulldozer (anchor 'steamroller', wordnet_cousin, wup 0.762); train (anchor 'steamroller', wordnet_cousin, wup 0.778)
- your verdict: fair / unfair because ...

## LVVIS/train_01706  (video_id 890, 29 frames)
- present: goose [base], swan [novel]
- easy absent: apricot, xylophone
- hard absent: seagull (anchor 'swan', wordnet_cousin, wup 0.815); chicken (anchor 'goose', wordnet_cousin, wup 0.889)
- your verdict: fair / unfair because ...

## LVVIS/train_02631  (video_id 1304, 15 frames)
- present: casserole [base], dining table [base], person [base], plate [base], spoon [novel]
- easy absent: stool, harmonium
- hard absent: spear (anchor 'spoon', wordnet_cousin, wup 0.75); tray (anchor 'plate', wordnet_cousin, wup 0.889)
- your verdict: fair / unfair because ...

## UVO/-F-aEPmjERo  (video_id 1926, 90 frames)
- present: bottle [base], dog [base], person [base], potted plant [novel], tv [novel]
- easy absent: ballet skirt, hippopotamus
- hard absent: camera (anchor 'tv', wordnet_cousin, wup 0.762); glass (anchor 'bottle', wordnet_cousin, wup 0.824)
- your verdict: fair / unfair because ...

## LVVIS/train_01597  (video_id 837, 14 frames)
- present: triangle [novel]
- easy absent: shaver, tachometer
- hard absent: sword (anchor 'triangle', wordnet_cousin, wup 0.8); target (anchor 'triangle', clip_fallback, wup 0.632)
- your verdict: fair / unfair because ...

## LVVIS/train_01457  (video_id 779, 24 frames)
- present: basketball backboard [base], cone [base], person [base], stepladder [novel], trash can [base], waist pack [novel], watch [novel]
- easy absent: gas pipe, eclair
- hard absent: sail (anchor 'cone', wordnet_cousin, wup 0.769); bowl (anchor 'cone', wordnet_cousin, wup 0.769)
- your verdict: fair / unfair because ...

## LVVIS/train_02604  (video_id 1294, 21 frames)
- present: bucket [base], dress [base], mango [novel], person [base]
- easy absent: kettlebell, table-tennis table
- hard absent: lemon (anchor 'mango', wordnet_cousin, wup 0.889); trash can (anchor 'bucket', wordnet_cousin, wup 0.778)
- your verdict: fair / unfair because ...
