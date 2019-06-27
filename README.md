[![PyPI](https://img.shields.io/pypi/v/prysm-api.svg)]()
[![](https://img.shields.io/badge/platform-Mac_Linux-red.svg)]()
[![](https://img.shields.io/badge/language-Python3-success.svg)](https://www.python.org/)
![GitHub](https://img.shields.io/github/license/fzhu2e/prysm-api.svg?color=blue)

# prysm-api
The API for [PRoxY System Modeling (PRYSM)](https://github.com/sylvia-dee/PRYSM).
Currently, it supports PSMs listed below:

+ Ice-core d18O
+ Tree-ring width with [VSLite](https://github.com/suztolwinskiward/VSLiteR)
+ Tree-ring cellulose
+ Tree MXD
+ Coral d18O and Sr/Ca
+ Speleothem d18O
+ Varve thickness

## How to install

Simply
```bash
pip install prysm-api LMRt  # LMRt provides many useful functions necessary for prysm-api
```

To use VS-Lite, in `R`
```bash
install.packages("devtools")
devtools::install_github("fzhu2e/VSLiteR")
```


## Usage examples

+ Calling VS-Lite

```python
import prysm

res = prysm.forward(
    'prysm.vslite',                 # psm name
    lat_obs, lon_obs,               # lat/lon of the target location
    lat, lon, time,                 # dimension variables of the environmentals
    {                               # environmental variables:
        'tas': tas,                 # surface air temperature in (time, lat, lon) [K]
        'pr': pr,                   # precipitation rate in (time, lat, lon) [kg/m2/s]
    },
    T1=8, T2=23, M1=0.01, M2=0.05,  # PSM specific parameters
)
```
Here `res` is a dictionary that includes the pseudoproxy values and the
corresponding timepoints, plus other output for diagnostics.

+ Calling the PSM for ice core d18O

```python
import prysm

res = prysm.forward(
    'prysm.ice.d18O',               # psm name
    lat_obs, lon_obs,               # lat/lon of the target location
    lat, lon, time,                 # dimension variables of the environmentals
    {                               # environmental variables:
        'tas': tas,                 # surface air temperature in (time, lat, lon) [K]
        'pr': pr,                   # precipitation rate in (time, lat, lon) [kg/m2/s]
        'psl': psl,                 # sea-level pressure in (time, lat, lon) [Pa]
        'd18Opr': d18Opr,           # precipitation d18O in (time, lat, lon) [permil]
    },
)
```

## References

+ Dee, S., J. Emile-Geay, M. N. Evans, A. Allam, E. J. Steig, and D. m. Thompson, 2015: PRYSM: An open-source framework for PRoxY System Modeling, with applications to oxygen-isotope systems. J. Adv. Model. Earth Syst., 7, 1220–1247, doi:10.1002/2015MS000447.
+ Dee, S. G., N. J. Steiger, J. Emile-Geay, and G. J. Hakim, 2016: On the utility of proxy system models for estimating climate states over the common era. J. Adv. Model. Earth Syst., 8, 1164–1179, doi:10.1002/2016MS000677.
+ Dee, S. G., L. A. Parsons, G. R. Loope, J. T. Overpeck, T. R. Ault, and J. Emile-Geay, 2017: Improved spectral comparisons of paleoclimate models and observations via proxy system modeling: Implications for multi-decadal variability. Earth and Planetary Science Letters, 476, 34–46, doi:10.1016/j.epsl.2017.07.036.
+ Dee, S. G., J. M. Russell, C. Morrill, Z. Chen, and A. Neary, 2018: PRYSM v2.0: A Proxy System Model for Lacustrine Archives. Paleoceanography and Paleoclimatology, 33, 1250–1269, doi:10.1029/2018PA003413.


## License

MIT License (see the details [here](LICENSE))
