#!/usr/bin/python
# -*- coding: utf-8 -*-

import datetime
import glob
import numpy as np
from netCDF4 import Dataset
import regionmask
import pytz
from scipy.interpolate import RectBivariateSpline
from skimage.morphology import remove_small_objects, binary_opening, disk, binary_closing
from skimage.measure import label, regionprops
import sys

REFERENCE_TIME = datetime.datetime(1949, 12, 1, tzinfo=pytz.utc)


def hrs_to_date(hrs):
    dt = datetime.timedelta(hours=int(hrs))
    return (REFERENCE_TIME+dt).date()


def date_to_hrs(date):
    if isinstance(date, datetime.date):
        date = datetime.datetime.combine(date, datetime.time(tzinfo=pytz.utc))
    dt = date-REFERENCE_TIME
    return int(dt.total_seconds()/3600)


class CycloneDetector(object):
    def __init__(self, filename):
        nc = Dataset(filename)
        self.debug_mode = 0.
        #print(self.debug_mode)
        self.dt = nc.variables["time"][1]-nc.variables["time"][0]
        self.xlon = nc.variables["xlon"][:]
        self.xlat = nc.variables["xlat"][:]
        N_lon = self.xlon.shape[1]
        N_lat = self.xlat.shape[0]
        self.idx_to_lon = RectBivariateSpline(np.arange(N_lon), np.arange(N_lat), self.xlon.T)
        self.idx_to_lat = RectBivariateSpline(np.arange(N_lon), np.arange(N_lat), self.xlat.T)
        self.left = self.xlon.min()
        self.right = self.xlon.max()
        self.bottom = self.xlat.min()+18
        self.top = self.xlat.max()-8
        land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
        ocean = np.logical_not(land.mask(self.xlon,self.xlat))
        self.ocean_mask = (ocean.data == 0)
        self.time_min = 0.
        self.time_max = 0.
        self.no_ingested = 0
        self.masks = np.empty((0,)+self.ocean_mask.shape)
        self.tc_table = np.empty((0,5))
        self.time = np.empty((0,))

    def detect_tc_in_step(self, nc, i, ecc_th=0.75):
        mask = self.ocean_mask.copy()
        uas = nc.variables["uas"][i].squeeze()
        vas = nc.variables["vas"][i].squeeze()
        wind_speed = np.sqrt(uas**2+vas**2)
        wind_mask = np.logical_and(self.ocean_mask, wind_speed > 20.)
        temp = nc.variables["ts"][i].squeeze()
        temp_mask = np.logical_and(self.ocean_mask, temp > 298.15)
        ps = nc.variables["ps"][i].squeeze()
        ps_mask = np.logical_and(self.ocean_mask, ps < 1005)
        mask = np.logical_or(wind_mask, np.logical_and(temp_mask, ps_mask))
        mask = remove_small_objects(mask, 20)
        lbl = label(mask)
        props_windspeed = regionprops(lbl, wind_speed)
        props_pressure = regionprops(lbl, ps)
        centroids = []
        for windspeed, pressure in zip(props_windspeed, props_pressure):
            max_wind_speed = windspeed["max_intensity"]
            min_pressure = pressure["min_intensity"]
            if windspeed["eccentricity"] > ecc_th or max_wind_speed<20.:
                lbl[lbl == windspeed["label"]]=0
            else:
                y, x = windspeed["centroid"]
                lon = float(self.idx_to_lon(x, y))
                lat = float(self.idx_to_lat(x, y))
                centroids.append([lon, lat, max_wind_speed, min_pressure])
        mask = lbl>0
        return mask, centroids

    def ingest_netcdf(self, filename):
        nc = Dataset(filename)
        time = nc.variables["time"][:]
        time_min = time.min()
        time_max = time.max()
        if time_min < self.time_max:
            raise RuntimeError("Non increasing times.")
        self.time = np.hstack([self.time, time])
        tc_table = []
        masks = np.empty(nc.variables["ps"].shape)
        for i in range(len(time)):
            masks[i,:,:], tcs = self.detect_tc_in_step(nc, i)
            for tc in tcs:
                row = [time[i]]
                row.extend(tc)
                tc_table.append(row)
        tc_table = np.array(tc_table)
        if self.no_ingested==0:
            self.time_min = time_min
        if self.debug_mode:
            self.masks = np.vstack([self.masks, masks])
        if len(tc_table)>0:
            self.tc_table = np.vstack([self.tc_table, tc_table])
        self.time_max = time_max
        self.no_ingested += 1

    def tracking(self, max_inactive_time=24., max_distance=6., min_length=4.):
        latest_track = 0
        active_tracks = []
        tracked_tcs = np.empty((self.tc_table.shape[0], self.tc_table.shape[1]+1))
        if len(self.tc_table)==0:
            self.tracked_tcs = tracked_tcs
        tracked_tcs[:,:-1] = self.tc_table
        tracked_tcs[:,-1] = 0.
        for i, (time, lon, lat, ws, mp) in enumerate(self.tc_table):
            new_active_tracks = []
            for at in active_tracks:
                lt, llon, llat, lws, lmp, ltno = tracked_tcs[tracked_tcs[:,-1]==at][-1]
                dt = time-lt
                if dt < max_inactive_time:
                    dist = np.sqrt((lon-llon)**2 + (lat-llat)**2)
                    if dist <= max_distance:
                        tracked_tcs[i,-1] = at
                    new_active_tracks.append(at)
            active_tracks = new_active_tracks
            if tracked_tcs[i,-1]==0:
                latest_track = tracked_tcs[i,-1] = latest_track + 1
                active_tracks.append(latest_track)
        lengths = np.bincount(tracked_tcs[:,-1].astype(int))
        mask = np.ones((len(tracked_tcs),), dtype='bool')
        for no, l in enumerate(lengths):
            if l < min_length:
                mask[tracked_tcs[:,-1]==no] = False
        tracked_tcs = tracked_tcs[mask]
        old_track_nos = set(tracked_tcs[:,-1])
        old_track_nos.discard(0)
        old_track_nos = sorted(old_track_nos)
        for i, o in enumerate(old_track_nos):
            tracked_tcs[tracked_tcs[:,-1]==o,-1] = i+1
        self.tracked_tcs = tracked_tcs

def build_filename_list():
    filenames = []
    for g in sys.argv[1:]:
        filenames.extend(glob.glob(g))
    return sorted(filenames)

def main():
    if len(sys.argv) < 2:
        print('Need input netCDF filenamei(s) with variables uas,vas,ts,ps')
        print('Example:')
        print(sys.argv[0]+' output/RegCM_SRF.2000*.nc')
        sys.exit(-1)
    filenames = build_filename_list()
    detector = CycloneDetector(filenames[0])
    for fn in filenames:
        detector.ingest_netcdf(fn)
    detector.tracking()
    tracked_tcs = detector.tracked_tcs
    # Save in a text file
    np.savetxt("tcs.dat", tracked_tcs)
    # and print on screen text file content
    with open("tcs.dat") as f:
        print(f.read( ))


if __name__ == "__main__":
    main()
