import json
from src.agent import _basic_metar_parse

# Test 1: LIFR & NO-GO conditions (Thunderstorms, 1500m visibility, cb clouds)
metar_lifr = "VECC 021100Z 22005KT 1500 TS SCT018 FEW025CB BKN090 31/24 Q1002"
print("=" * 65)
print("🔍 TEST 1: LIFR & NO-GO WEATHER (Thunderstorms, 1500m visibility, CB)")
print("=" * 65)
result_lifr = _basic_metar_parse(metar_lifr)
print(json.dumps(result_lifr, indent=2))

# Test 2: VFR & GO conditions (Clear skies, light winds)
metar_vfr = "KJFK 151251Z 18005KT 10SM FEW025 22/15 A3002"
print("\n" + "=" * 65)
print("🔍 TEST 2: VFR & GO WEATHER (Clear skies, light winds)")
print("=" * 65)
result_vfr = _basic_metar_parse(metar_vfr)
print(json.dumps(result_vfr, indent=2))

# Test 3: IFR & NO-GO conditions (High winds, gust spread, low ceiling)
metar_ifr = "KORD 151451Z 32015G28KT 6SM -RA BR OVC008 12/11 A2975"
print("\n" + "=" * 65)
print("🔍 TEST 3: IFR & NO-GO WEATHER (28KT Gusts, 800ft Ceiling, Rain)")
print("=" * 65)
result_ifr = _basic_metar_parse(metar_ifr)
print(json.dumps(result_ifr, indent=2))
