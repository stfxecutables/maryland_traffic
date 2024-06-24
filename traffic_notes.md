# Runtimes

## Full Data

```
TARGET=outcome
DROPS='search_conducted,search_disposition,search_type'
```

~15 minutes for type inference and univariate metrics
~30 minutes to tune first embedded model (I think LGBM?)
~24 hours to tune second embedded model (Linear?)

Observations:


```
### Continuous Features (Accuracy: Higher = More important)

| feature         |       acc |
|:----------------|----------:|
| hour_of_stop    | 1.200e-02 |
| year_of_stop    | 7.333e-03 |
| license_lat     | 1.333e-03 |
| vehicle_year    | 0.000e+00 |
|       ...       |    ...    |

### Categorical Features (Accuracy: Higher = More important)

| feature           |       acc |
|:------------------|----------:|
| violation_type    | 3.627e-01 |
| chrg_title        | 1.140e-01 |
| chrg_sect_mtch    | 9.200e-02 |
| chrg_title_mtch   | 5.933e-02 |
| stop_chrg_title   | 4.867e-02 |
| article           | 2.600e-02 |
| race              | 2.267e-02 |
| accident          | 1.333e-02 |
| acc_blame         | 1.333e-02 |
| prop_dmg          | 6.000e-03 |
| unmarked_arrest   | 4.667e-03 |
| pers_injury       | 4.000e-03 |
| licensed_outstate | 4.000e-03 |
| comm_license      | 4.000e-03 |
| belts             | 3.333e-03 |
| patrol_entity     | 2.667e-03 |
| vehicle_make      | 2.000e-03 |
| sex               | 0.000e+00 |
|        ...        |    ...    |

```

Above suggests `charge` and `violation_type` should be removed as features
(are results, not causes). This also means charge-derived features (eg.
"chrg_*_mtch" features) are suspect for leakage. This also includes the
"article" feature.

# Features

subagency: categorical
  - '3rd District, Silver Spring'
  - '6th District, Gaithersburg / Montgomery Village'
  - '4th District, Wheaton'
  - '2nd District, Bethesda'
  - 'Headquarters and Special Operations'
  - '5th District, Germantown'
  - '1st District, Rockville'
  - 'W15'
  - 'S15'
latitude: continuous
  - latitude of stop
longitude: continuous
  - longitude of stop
accident: bool
  - is accident or not
belts: bool
  - seatbelts worn / present
pers_injury: bool
  - any person injured
prop_dmg: bool
  - any property damage
fatal: bool
  - any fatalities
comm_license: bool
  - commercial license
hazmat: bool
  - ???
comm_vehicle: bool
  - commercial vehicle
alcohol: bool
  - alcohol involved
work_zone: bool
  - stop in a work zone
search_conducted: bool
  - if search conducted on stop
search_disposition: categorical
  - "nan", "Property Only", None, "Contraband Only", "Contraband and Property"
search_type: categorical
  - 'nan', 'Both', 'Person', 'Property', None
vehicle_type: categorical
  - auto, truck, other, cycle, trailer, bus, farm, etc., 10 categories
vehicle_year: ordinal
  - year of vehicle
vehicle_make: categorical
  - chevy, honda, toyota, etc.
vehicle_color: categorical
  - paint color
violation_type: categorical
  - Citation, ESERO, SERO, Warning
article: categorical
  - 'Maryland Rules', 'Transportation Article', 'nan', 'other'
acc_blame: bool
  - accept blame? >97% are true, not a very useful feature
race: categorical
  - ancestry
sex: categorical
  - biological sex ("M", "F", "U")
patrol_entity: categorical
  - e.g. marked patrol, unmarked patrol, marked laser, etc.
hour_of_stop: continuous
  - 0-24.0
year_of_stop: ordinal
  - 2023, ...
month_of_stop: ordinal
  - 0-11
weeknum_of_stop: ordinal
  - week of year
weekday_of_stop: ordinal
  - day of week
reg_lat: continuous
  - registration latitude
reg_long: continuous
  - registration longitude
reg_km: continuous
  - distance between vehicle registration location and stop/search location
home_lat: continuous
  - home latitude (converted from state)
home_long: continuous
  - home longitude (converted from state)
license_lat: continuous
  - license latitude
license_long: continuous
  - license longitude
chrg_title: ordinal
  - charge title number (suspected charge / stop reason)
stop_chrg_title: ordinal
  - stop actual charge title number
chrg_title_mtch: bool
  - if suspected charge matches stop charge
chrg_sect_mtch: bool
  - if suspected charge section number matches stop charge section number
unmarked_arrest: bool
  - if arrest was unmarked
tech_arrest: bool
  - if arrest used technology only (e.g. laser, camera)
home_outstate: bool
  - if driver lives out of state
licensed_outstate: bool
  - if driver is licensed out of state

outcome: categorical [TARGET]
  - 'Citation', 'Arrest', 'nan', 'Warning', 'SERO', None

## Drops (Invalid Predictors)

There are a number of highly-related categorical variables that can each
function as a target (i.e. are **caused outcomes**), and so should never be
included in the same model as features. The Cramer V correlations are:

                   final   stop    charge   charge
                   charge  charge  section  title   search  search          viol-
                   title   title   matches  matches condu-  dispos  search  ation
                                   stop     stop    -cted   ion     type    type   article

outcome             0.16   0.12    0.61     0.70    0.82    0.50    0.49    0.52   0.35
article             0.71   0.46    0.21     0.16    0.03    0.02    0.02    0.58
violation_type      0.34   0.14    0.28     0.20    0.16    0.07    0.09
search_type         0.04   0.04    0.03     0.08    1.00    0.61
search_disposition  0.03   0.04    0.03     0.06    1.00
search_conducted    0.06   0.05    0.03     0.08
chrg_title_mtch     0.29   0.15    0.80
chrg_sect_mtch      0.27   0.08
stop_chrg_title     0.60



search_conducted: bool
  - if search conducted on stop
search_disposition: categorical
  - "nan", "Property Only", None, "Contraband Only", "Contraband and Property"
search_type: categorical
  - 'nan', 'Both', 'Person', 'Property', None

chrg_title: ordinal
  - charge title number (suspected charge / stop reason)
stop_chrg_title: ordinal
  - stop actual charge title number
chrg_title_mtch: bool
  - if suspected charge matches stop charge
chrg_sect_mtch: bool
  - if suspected charge section number matches stop charge section number

violation_type: categorical
  - Citation, ESERO, SERO, Warning
outcome: categorical [TARGET]
  - 'Citation', 'Arrest', 'nan', 'Warning', 'SERO', None

### Unhelpful

article: categorical
  - 'Maryland Rules', 'Transportation Article', 'nan', 'other'
acc_blame: bool
  - accept blame? >97% are true, not a very useful feature


## Targets

search_conducted: bool
  - if search conducted on stop
search_disposition: categorical
  - "nan", "Property Only", None, "Contraband Only", "Contraband and Property"
search_type: categorical
  - 'nan', 'Both', 'Person', 'Property', None

chrg_title_mtch: bool
  - if suspected charge matches stop charge
chrg_sect_mtch: bool
  - if suspected charge section number matches stop charge section number


# Runs

1. TARGET=outcome
1. TARGET=violation_type
1. TARGET=search_conducted
1. TARGET=chrg_title_mtch