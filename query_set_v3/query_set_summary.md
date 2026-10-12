# Query set summary (v3)

- query_set.json SHA-256: `6d08f1a2408ade15446917896f7d053527263de291cea4e3dfc4b14342e27713`
- embedder: **clip**
- videos in set: 1972 (usable 1972, skipped 0)

## WordNet resolution of category names

- exact name: 960 | head noun only (approximate): 77 | unresolved: 16 | coverage 98.5%
- full mapping: `category_wordnet_map.csv`
- head-noun (approximate, never used as hard prompts): action camera, alcohol lamp, balance scale, beluga whale, binder clip, board eraser, body thermometer, brass plaque, canvas bag, car jack, cell phone charger, cheese curls, chef hat, computer box, correction fluid, correction tape, cotton swab, double-sided tape, drum set, e-reader, egg tart, electric bicycle, electric kettle, flash drive, flying disc, folding knife, fridge magnet, fried chicken, fume hood, game console, gas pipe, green plants, halter top, hand grips strengthener, hardback book, high heels, high jump standards, horse buggy, incense burner, inflatable bed, infusion pump, kitchen paper, leather shoes, long jump pit, magic cube, mechanical pencil, memo pad, pasta strainer, picnic basket, potted plant, power bank, protective suit, relay baton, remote control car, rice cooker, roast duck, safety hammer, sd card, selfie stick, side table, stirring rod, stuffed animal, swim cap, swim ring, swimming goggles, test tube holder, travel pillow, virtual reality headset, water temperature gauge, weightlifting belt, welding torch, white sugar, wireless chargers, wood plane, writing brush, xylophone mallets, yoga mat
- unresolved: barcode, billards, cheongsam, dehumidifier, dragonfruit, gamepad, hoodie, humidifier, kettlebell, nightstand, oxygen concentrator, ramen, smartwatch, soundbar, starfruit, whiteboard
- manual exclusion pairs: 45 (sha256 c7b84039c35f948dc92dbaf68cecd3afed2a5448537d604d159b95864d3134ba)

## Reference pairs (is the cousin threshold sensible?)

Threshold: Wu-Palmer >= 0.75. Parent/child pairs are always excluded.

| pair | Wu-Palmer | parent/child? |
|---|---:|---|
| goose / duck | 0.929 | False |
| dolphin / seal | 0.8 | False |
| car / truck | 0.917 | False |
| bowl / pot | 0.889 | False |
| dog / bear | 0.889 | False |
| dog / cat | 0.857 | False |
| person / cat | 0.706 | True |
| person / doll | 0.667 | True |
| bottle / alcohol | 0.25 | False |
| knob / ball | 0.714 | False |

## How hard prompts were chosen

- clip_fallback: 145
- wordnet_cousin: 3799

## Query counts

| type | total | base | novel |
|---|---:|---:|---:|
| present | 7808 | 4728 | 3080 |
| easy_absent | 3944 | 1755 | 2189 |
| hard_absent | 3944 | 2548 | 1396 |

## Diversity check: most frequent prompts (one word dominating = bad)

- **hard_absent**: cup (97), plate (63), bait (60), knife (60), toy (55), fork (52), fish (51), car (50), horse (47), spear (45), level (43), box (43), dog (41), alcohol (41), camera (39)
- **easy_absent**: asparagus (20), needle (17), propeller (16), cuttlefish (16), tassel (16), pool table (15), trowel (15), table-tennis table (14), starfruit (14), pool cue (14), sparrow (14), headband (14), ironing board (14), fireplug (14), wet suit (14)
- **hard anchors**: person (105), bird (87), watch (59), bottle (53), jersey (47), trousers (43), knife (38), shoe (33), ring (31), hat (31)

- videos with fewer than 2+2 absent prompts: 0

## Candidates removed by fairness filters (summed over videos)

- manual_exclusion: 752
- near_synonym_by_clip: 1127
- same_prompt_text: 15
- whole_word_overlap: 3955
- wordnet_parent_child_or_same: 173101
