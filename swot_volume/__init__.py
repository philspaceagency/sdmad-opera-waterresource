"""Water level and storage volume of a reservoir from SWOT lake heights and Sentinel-1 water area.

    swot.py        search / download SWOT L2_HR_LakeSP (earthaccess) and read the lake polygons that cover a reservoir
    hypsometry.py  area-elevation curve A(h) from (SWOT level, Sentinel-1 area) pairs, and volume V(h) by integrating it
    integrate.py   pair the two series in time and produce level, area and volume per date
"""
