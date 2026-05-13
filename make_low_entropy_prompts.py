"""
Generate 500 low-entropy prompts across structured/factual categories.
Saves to data/low_entropy_prompts.jsonl — one {"prompt": "..."} per line.

Usage:
    python make_low_entropy_prompts.py
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Prompt templates by category
# ---------------------------------------------------------------------------

COUNTRIES = [
    ("France","Paris"),("Germany","Berlin"),("Italy","Rome"),("Spain","Madrid"),
    ("Japan","Tokyo"),("China","Beijing"),("Brazil","Brasilia"),("India","New Delhi"),
    ("Canada","Ottawa"),("Australia","Canberra"),("Russia","Moscow"),("Mexico","Mexico City"),
    ("Argentina","Buenos Aires"),("South Korea","Seoul"),("Egypt","Cairo"),
    ("Nigeria","Abuja"),("South Africa","Pretoria"),("Turkey","Ankara"),
    ("Saudi Arabia","Riyadh"),("Indonesia","Jakarta"),("Portugal","Lisbon"),
    ("Netherlands","Amsterdam"),("Sweden","Stockholm"),("Norway","Oslo"),
    ("Denmark","Copenhagen"),("Finland","Helsinki"),("Poland","Warsaw"),
    ("Austria","Vienna"),("Switzerland","Bern"),("Belgium","Brussels"),
    ("Greece","Athens"),("Hungary","Budapest"),("Czech Republic","Prague"),
    ("Romania","Bucharest"),("Ukraine","Kyiv"),("Thailand","Bangkok"),
    ("Vietnam","Hanoi"),("Philippines","Manila"),("Malaysia","Kuala Lumpur"),
    ("Pakistan","Islamabad"),("Bangladesh","Dhaka"),("Iran","Tehran"),
    ("Iraq","Baghdad"),("Israel","Jerusalem"),("Jordan","Amman"),
    ("Kenya","Nairobi"),("Ethiopia","Addis Ababa"),("Ghana","Accra"),
    ("Morocco","Rabat"),("Algeria","Algiers"),
]

ELEMENTS = [
    ("hydrogen","H"),("helium","He"),("lithium","Li"),("carbon","C"),
    ("nitrogen","N"),("oxygen","O"),("fluorine","F"),("neon","Ne"),
    ("sodium","Na"),("magnesium","Mg"),("aluminum","Al"),("silicon","Si"),
    ("phosphorus","P"),("sulfur","S"),("chlorine","Cl"),("argon","Ar"),
    ("potassium","K"),("calcium","Ca"),("iron","Fe"),("copper","Cu"),
    ("zinc","Zn"),("silver","Ag"),("gold","Au"),("mercury","Hg"),
    ("lead","Pb"),("uranium","U"),("platinum","Pt"),("nickel","Ni"),
]

MATH = [
    ("2 + 2","4"),("3 × 3","9"),("10 - 4","6"),("15 ÷ 3","5"),
    ("5 × 5","25"),("100 - 1","99"),("7 + 8","15"),("4 × 6","24"),
    ("2 × 2 × 2","8"),("12 ÷ 4","3"),("9 × 9","81"),("6 + 7","13"),
    ("20 - 8","12"),("3 + 4 + 5","12"),("11 × 11","121"),
]

SEQUENCES = [
    "Monday, Tuesday, Wednesday, Thursday,",
    "January, February, March, April, May,",
    "spring, summer, autumn,",
    "Mercury, Venus, Earth, Mars,",
    "1st, 2nd, 3rd, 4th,",
    "do, re, mi, fa, sol,",
    "alpha, beta, gamma, delta,",
    "north, south, east,",
    "red, orange, yellow, green, blue,",
    "one, two, three, four,",
    "A, B, C, D, E,",
    "first, second, third,",
    "breakfast, lunch,",
    "morning, afternoon,",
    "penny, nickel, dime,",
]

FACTUAL = [
    "Water freezes at 0 degrees Celsius and boils at",
    "The speed of light in a vacuum is approximately",
    "The human body has",
    "DNA stands for",
    "The Earth orbits the Sun once every",
    "The Great Wall of China is located in",
    "William Shakespeare was born in",
    "The Eiffel Tower is located in",
    "The Amazon River is located in",
    "Mount Everest is the tallest mountain in the",
    "The Pythagorean theorem states that a squared plus b squared equals",
    "The first man to walk on the moon was",
    "Albert Einstein developed the theory of",
    "The chemical formula for water is",
    "The capital city of the United States is",
    "The largest ocean on Earth is the",
    "The sum of angles in a triangle is",
    "Photosynthesis converts sunlight into",
    "The heart pumps",
    "Isaac Newton discovered the law of",
]

STRUCTURED = [
    "Name: Alice\nAge: 30\nOccupation:",
    "Product: Laptop\nPrice: $999\nBrand:",
    "City: New York\nCountry: USA\nPopulation:",
    "Author: George Orwell\nPublished: 1949\nTitle:",
    "Language: Python\nVersion: 3.11\nType:",
    "Planet: Mars\nMoons: 2\nDistance from Sun:",
    "Animal: Eagle\nClass: Bird\nHabitat:",
    "Dish: Pizza\nOrigin: Italy\nMain ingredient:",
    "Sport: Tennis\nPlayers per side: 1\nEquipment:",
    "Currency: Dollar\nCountry: USA\nSymbol:",
]

QUOTES = [
    "To be or not to be, that is the",
    "Four score and seven years ago our",
    "Ask not what your country can do for you, ask what",
    "I have a dream that one day",
    "In the beginning God created the",
    "It was the best of times, it was the worst of",
    "We hold these truths to be self-evident, that all men are created",
    "The only thing we have to fear is",
    "E pluribus",
    "Veni, vidi,",
]

# ---------------------------------------------------------------------------
# Build prompt list
# ---------------------------------------------------------------------------

prompts = []

# Country capitals — multiple phrasings
for country, capital in COUNTRIES:
    prompts.append(f"The capital of {country} is")
    prompts.append(f"Q: What is the capital city of {country}?\nA:")
    prompts.append(f"Country: {country}\nCapital:")

# Elements
for element, symbol in ELEMENTS:
    prompts.append(f"The chemical symbol for {element} is")
    prompts.append(f"Element: {element}\nSymbol:")

# Math
for expr, answer in MATH:
    prompts.append(f"{expr} =")
    prompts.append(f"What is {expr}? The answer is")

# Sequences
for seq in SEQUENCES:
    prompts.append(seq)
    prompts.append(f"Complete the sequence: {seq}")

# Factual
for fact in FACTUAL:
    prompts.append(fact)

# Structured
for s in STRUCTURED:
    prompts.append(s)

# Quotes
for q in QUOTES:
    prompts.append(q)

# Trim or pad to exactly 500
import random
random.seed(42)
if len(prompts) > 500:
    prompts = random.sample(prompts, 500)
else:
    # Cycle through to fill to 500
    while len(prompts) < 500:
        prompts.append(random.choice(prompts[:len(prompts)]))

random.shuffle(prompts)
prompts = prompts[:500]

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

Path("data").mkdir(exist_ok=True)
out = Path("data/low_entropy_prompts.jsonl")
with open(out, "w") as f:
    for i, p in enumerate(prompts):
        f.write(json.dumps({"idx": i, "prompt": p}) + "\n")

print(f"Saved {len(prompts)} prompts to {out}")
for cat, ex in [
    ("Country capital", f"The capital of {COUNTRIES[0][0]} is"),
    ("Element",         f"The chemical symbol for {ELEMENTS[0][0]} is"),
    ("Math",            f"{MATH[0][0]} ="),
    ("Sequence",        SEQUENCES[0]),
    ("Factual",         FACTUAL[0]),
    ("Structured",      STRUCTURED[0].split('\n')[0]),
    ("Quote",           QUOTES[0]),
]:
    print(f"  [{cat}] {ex!r}")
