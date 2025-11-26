# cleaning.py
# Full, version-aware AIS cleaning pipeline for MarineCadastre AIS (2018–2024 and 2025+)
# Dependencies: pandas, numpy; optional: shapely, geopandas (for AOI/landmask)
import pandas as pd
import numpy as np
from pathlib import Path

# ---------- Optional shapefile loader ----------
def load_union_polygon(path):
    import geopandas as gpd
    gdf = gpd.read_file(path)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf.union_all

# ---------- Utils ----------
def _to_utc(ts):
    return pd.to_datetime(ts, utc=True, errors='coerce')

def _wrap360(a):
    return np.mod(a, 360.0)

def _ang_diff_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0

def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 6371.0088 * (2.0 * np.arcsin(np.sqrt(a)))

def _validate_mmsi_series(mmsi_s):
    s = pd.to_numeric(mmsi_s, errors='coerce').astype('Int64').astype(str).str.strip()
    valid_len = s.str.len() == 9
    digits = s.str.isnumeric()
    first_ok = s.str[0].isin(list("234567"))
    ok = valid_len & digits & first_ok
    out = pd.to_numeric(s.where(ok), errors='coerce')
    return out

def _len_based_max_speed_kn(length_arr, default_kn=60.0):
    arr = np.asarray(length_arr, dtype=float)  # NaN-safe
    res = np.full(arr.shape, float(default_kn))
    m200 = arr > 200
    m100 = (arr > 100) & (arr <= 200)
    m50  = (arr > 50)  & (arr <= 100)
    mlt  = (arr > 0)   & (arr <= 50)
    res[m200] = 30.0
    res[m100] = 26.0
    res[m50]  = 22.0
    res[mlt]  = 18.0
    return res

# ---------- Config ----------
class CleanConfig:
    def __init__(self,
                 context="port",             # "port" or "offshore"
                 aoi_polygon=None,           # shapely Polygon/MultiPolygon or path to .shp
                 aoi_buffer_km=50,
                 landmask_polygon=None,      # shapely Polygon/MultiPolygon or path to .shp
                 gap_threshold_s_port=1800,  # 30 min
                 gap_threshold_s_off=7200,   # 2 h
                 max_speed_kn_default=60.0,
                 max_speed_kn_port=20.0,
                 jump_dist_km=10.0,
                 jump_dt_s=120,
                 sog_spike_kn=20.0,
                 sog_spike_dt_s=60,
                 turn_min_sog_kn=15.0,
                 turn_rate_max_deg_per_min=40.0,
                 downsample_cadence_s=60,
                 enforce_domains=True,
                 round_to_resolution=True):
        self.context = context
        self.aoi_polygon = aoi_polygon
        self.aoi_buffer_km = aoi_buffer_km
        self.landmask_polygon = landmask_polygon
        self.gap_threshold_s_port = gap_threshold_s_port
        self.gap_threshold_s_off = gap_threshold_s_off
        self.max_speed_kn_default = max_speed_kn_default
        self.max_speed_kn_port = max_speed_kn_port
        self.jump_dist_km = jump_dist_km
        self.jump_dt_s = jump_dt_s
        self.sog_spike_kn = sog_spike_kn
        self.sog_spike_dt_s = sog_spike_dt_s
        self.turn_min_sog_kn = turn_min_sog_kn
        self.turn_rate_max_deg_per_min = turn_rate_max_deg_per_min
        self.downsample_cadence_s = downsample_cadence_s
        self.enforce_domains = enforce_domains
        self.round_to_resolution = round_to_resolution

# ---------- Cleaner ----------
class AISCleaner:
    def __init__(self, config: CleanConfig):
        # Auto-load polygons if string paths are provided
        if isinstance(config.landmask_polygon, (str, Path)):
            config.landmask_polygon = load_union_polygon(config.landmask_polygon)
        if isinstance(config.aoi_polygon, (str, Path)):
            config.aoi_polygon = load_union_polygon(config.aoi_polygon)
        self.cfg = config
        self.audit = {}

    def _harmonize_schema(self, df):
        # Map 2018–2024 vs 2025+ names into a unified schema [1]
        rename = {
            'mmsi':'MMSI', 'MMSI':'MMSI',
            'base_date_time':'BaseDateTime','BaseDateTime':'BaseDateTime',
            'latitude':'LAT','LAT':'LAT',
            'longitude':'LON','LON':'LON',
            'sog':'SOG','SOG':'SOG',
            'cog':'COG','COG':'COG',
            'heading':'Heading','Heading':'Heading',
            'length':'Length','Length':'Length',
            'width':'Width','Width':'Width',
            'draft':'Draft','Draft':'Draft',
            'vessel_type':'VesselType','VesselType':'VesselType',
            'status':'Status','Status':'Status',
            'cargo':'Cargo','Cargo':'Cargo',
            'vessel_name':'VesselName','VesselName':'VesselName',
            'imo':'IMO','IMO':'IMO',
            'call_sign':'CallSign','CallSign':'CallSign',
            'transceiver':'TransceiverClass','TransceiverClass':'TransceiverClass'
        }
        df = df.rename(columns={c: rename.get(c, c) for c in df.columns})

        # Types: floats for numeric computations (avoid pd.NA in comparisons)
        df['MMSI'] = _validate_mmsi_series(df.get('MMSI'))
        df['BaseDateTime'] = _to_utc(df.get('BaseDateTime'))
        for c in ['LAT','LON','SOG','COG','Heading','Length','Width','Draft','VesselType','Status','Cargo']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        for c in ['VesselName','IMO','CallSign','TransceiverClass']:
            if c in df.columns:
                df[c] = df[c].astype('string')

        # Normalize transceiver to A/B
        if 'TransceiverClass' in df.columns:
            df['TransceiverClass'] = df['TransceiverClass'].str.upper().str.strip()
            df.loc[~df['TransceiverClass'].isin(['A','B']), 'TransceiverClass'] = pd.NA
        return df

    def load_and_normalize(self, paths_or_df):
        if isinstance(paths_or_df, pd.DataFrame):
            raw = paths_or_df.copy()
        else:
            paths = paths_or_df if isinstance(paths_or_df, (list, tuple)) else [paths_or_df]
            frames = []
            for p in paths:
                ext = str(p).lower()
                if ext.endswith(".parquet"):
                    frames.append(pd.read_parquet(p))
                else:
                    frames.append(pd.read_csv(p))
            raw = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        df = self._harmonize_schema(raw)

        # Required non-null fields
        req_mask = (~df[['MMSI','BaseDateTime','LAT','LON']].isna()).all(axis=1)
        self.audit['required_nulls'] = df.loc[~req_mask]
        df = df.loc[req_mask].copy()
        # Memory-friendly types
        df['MMSI'] = df['MMSI'].astype('int32')
        return df

    def enforce_domains(self, df):
        if not self.cfg.enforce_domains:
            self.audit['domains'] = pd.DataFrame(columns=df.columns)
            return df
        df = df.copy()
        # Position bounds [1]
        pos_ok = df['LAT'].between(-89.99999, 89.99999) & df['LON'].between(-179.99999, 179.99999)
        removed = df.loc[~pos_ok]
        self.audit['domains'] = removed
        df = df.loc[pos_ok].copy()
        # Kinematics and dimensions [1]
        if 'SOG' in df.columns: df['SOG'] = df['SOG'].clip(0, 99.9)
        if 'COG' in df.columns: df['COG'] = _wrap360(df['COG'].clip(0, 359.9))
        if 'Heading' in df.columns: df['Heading'] = _wrap360(df['Heading'].clip(0, 359))
        if 'Length' in df.columns: df['Length'] = df['Length'].where(df['Length'].between(1, 509), np.nan)
        if 'Width'  in df.columns: df['Width']  = df['Width'].where(df['Width'].between(1, 61), np.nan)
        if 'Draft'  in df.columns: df['Draft']  = df['Draft'].where(df['Draft'].between(1, 24), np.nan)
        # Codes [1]
        if 'VesselType' in df.columns: df['VesselType'] = df['VesselType'].where(df['VesselType'].between(1, 1024), np.nan)
        if 'Status'     in df.columns: df['Status']     = df['Status'].where(df['Status'].between(1, 14), np.nan)
        if 'Cargo'      in df.columns: df['Cargo']      = df['Cargo'].where(df['Cargo'].between(1, 1024), np.nan)
        return df

    def round_to_resolution(self, df):
        if not self.cfg.round_to_resolution:
            self.audit['resolution_rounding'] = pd.DataFrame(columns=df.columns)
            return df
        df = df.copy()
        for c in ['SOG','COG']:
            if c in df.columns:
                df[c] = (df[c] * 10.0).round().div(10.0)
        for c in ['Length','Width']:
            if c in df.columns:
                df[c] = df[c].round()
        if 'Draft' in df.columns:
            df['Draft'] = (df['Draft'] * 10.0).round().div(10.0)
        self.audit['resolution_rounding'] = pd.DataFrame(columns=df.columns)
        return df

    def basic_hygiene(self, df):
        df = df.sort_values(['MMSI','BaseDateTime']).copy()
        dupe = df.duplicated(subset=['MMSI','BaseDateTime'], keep='last')
        self.audit['dedupe_mmsi_time'] = df.loc[dupe]
        return df.loc[~dupe].copy()

    def geofence_aoi(self, df):
        if self.cfg.aoi_polygon is None:
            self.audit['geofence_aoi'] = pd.DataFrame(columns=df.columns)
            return df
        try:
            from shapely.geometry import Point
            poly = self.cfg.aoi_polygon
            mean_lat = df['LAT'].mean()
            deg_per_km_lat = 1.0 / 111.32
            deg_per_km_lon = 1.0 / (111.32 * np.cos(np.radians(mean_lat)))
            buff_lat = self.cfg.aoi_buffer_km * deg_per_km_lat
            buff_lon = self.cfg.aoi_buffer_km * deg_per_km_lon
            minx, miny, maxx, maxy = poly.bounds
            minx -= buff_lon; maxx += buff_lon
            miny -= buff_lat; maxy += buff_lat
            bbox = df['LON'].between(minx, maxx) & df['LAT'].between(miny, maxy)
            pre = df.loc[bbox].copy()
            pts = pre[['LON','LAT']].to_numpy()
            contains = np.fromiter((poly.contains(Point(x, y)) for x, y in pts), dtype=bool, count=len(pts))
            kept = pre[contains]
            self.audit['geofence_aoi'] = df.loc[~df.index.isin(kept.index)]
            return kept
        except Exception:
            self.audit['geofence_aoi'] = pd.DataFrame(columns=df.columns)
            return df

    def landmask_filter(self, df):
        if self.cfg.landmask_polygon is None:
            self.audit['landmask'] = pd.DataFrame(columns=df.columns)
            return df
        try:
            from shapely.geometry import Point
            land = self.cfg.landmask_polygon
            pts = df[['LON','LAT']].to_numpy()
            on_land = np.fromiter((land.contains(Point(x, y)) for x, y in pts), dtype=bool, count=len(pts))
            removed = df.loc[on_land]
            self.audit['landmask'] = removed
            return df.loc[~on_land].copy()
        except Exception:
            self.audit['landmask'] = pd.DataFrame(columns=df.columns)
            return df

    def sort_and_segment(self, df):
        df = df.sort_values(['MMSI','BaseDateTime']).copy()
        dt = df.groupby('MMSI')['BaseDateTime'].diff().dt.total_seconds()
        gap_thr = self.cfg.gap_threshold_s_port if self.cfg.context == 'port' else self.cfg.gap_threshold_s_off
        new_seg = (dt.isna()) | (dt > gap_thr) | (dt <= 0)
        df['segment_id'] = new_seg.groupby(df['MMSI']).cumsum()
        self.audit['segmenting'] = pd.DataFrame(columns=df.columns)
        return df

    def kinematic_checks(self, df):
        grp = ['MMSI','segment_id']
        lat_prev = df.groupby(grp)['LAT'].shift(1)
        lon_prev = df.groupby(grp)['LON'].shift(1)
        t_prev = df.groupby(grp)['BaseDateTime'].shift(1)
        dt_s = (df['BaseDateTime'] - t_prev).dt.total_seconds()
        dist_km = _haversine_km(lat_prev.values, lon_prev.values, df['LAT'].values, df['LON'].values)
        spd_kn = dist_km / (dt_s / 3600.0)
        df['pos_speed_kn'] = spd_kn

        if 'Length' in df.columns:
            max_kn_by_len = _len_based_max_speed_kn(df['Length'].to_numpy(), self.cfg.max_speed_kn_default)
        else:
            max_kn_by_len = np.full(len(df), float(self.cfg.max_speed_kn_default))
        max_kn_ctx = self.cfg.max_speed_kn_port if self.cfg.context == 'port' else self.cfg.max_speed_kn_default
        max_allowed = np.minimum(max_kn_by_len, max_kn_ctx)

        invalid = (dt_s <= 0) | ((~np.isnan(spd_kn)) & (spd_kn > max_allowed))
        removed = df.loc[invalid]
        self.audit['kinematic_checks'] = removed
        return df.loc[~invalid].copy()

    def jumps_and_spikes(self, df):
        grp = ['MMSI','segment_id']
        t_prev = df.groupby(grp)['BaseDateTime'].shift(1)
        dt_s = (df['BaseDateTime'] - t_prev).dt.total_seconds()
        lat_prev = df.groupby(grp)['LAT'].shift(1)
        lon_prev = df.groupby(grp)['LON'].shift(1)
        dist_km = _haversine_km(lat_prev.values, lon_prev.values, df['LAT'].values, df['LON'].values)
        jump_mask = (dt_s <= self.cfg.jump_dt_s) & (dist_km >= self.cfg.jump_dist_km)

        sog_prev = df.groupby(grp)['SOG'].shift(1) if 'SOG' in df.columns else pd.Series(np.nan, index=df.index)
        sog_delta = np.abs(df['SOG'] - sog_prev) if 'SOG' in df.columns else pd.Series(False, index=df.index)
        spike_mask = (dt_s <= self.cfg.sog_spike_dt_s) & (sog_delta > self.cfg.sog_spike_kn)

        removed_mask = jump_mask | spike_mask
        self.audit['jumps_spikes'] = df.loc[removed_mask]
        return df.loc[~removed_mask].copy()

    def turn_rate_sanity(self, df):
        if 'COG' not in df.columns:
            self.audit['turn_rate'] = pd.DataFrame(columns=df.columns)
            return df
        grp = ['MMSI','segment_id']
        cog_prev = df.groupby(grp)['COG'].shift(1)
        t_prev = df.groupby(grp)['BaseDateTime'].shift(1)
        dt_min = (df['BaseDateTime'] - t_prev).dt.total_seconds() / 60.0
        turn_rate = np.abs(_ang_diff_deg(df['COG'].values, cog_prev.values)) / dt_min
        sog = df['SOG'] if 'SOG' in df.columns else pd.Series(np.nan, index=df.index)
        bad_turn = (sog > self.cfg.turn_min_sog_kn) & (turn_rate > self.cfg.turn_rate_max_deg_per_min)
        self.audit['turn_rate'] = df.loc[bad_turn]
        return df.loc[~bad_turn].copy()

    def clone_mmsi_resolution(self, df):
        key = ['MMSI','BaseDateTime']
        dup_idx = df.duplicated(subset=key, keep=False)
        df_dup = df.loc[dup_idx].copy()
        if df_dup.empty:
            self.audit['clone_resolution'] = pd.DataFrame(columns=df.columns)
            return df
        cx, cy = df['LON'].median(), df['LAT'].median()
        keep_idx, remove_idx = [], []
        for (_, _), g in df_dup.groupby(key):
            dkm = _haversine_km(g['LAT'].values, g['LON'].values, np.full(len(g), cy), np.full(len(g), cx))
            j = int(np.argmin(dkm))
            keep_idx.append(g.index[j])
            remove_idx.extend([idx for i, idx in enumerate(g.index) if i != j])
        self.audit['clone_resolution'] = df.loc[remove_idx]
        return df.loc[~df.index.isin(remove_idx)].copy()

    def downsample(self, df):
        cadence = pd.Timedelta(seconds=self.cfg.downsample_cadence_s)
        bucket = (df['BaseDateTime'].astype('int64') // cadence.value) * cadence.value
        df = df.copy()
        df['bucket'] = pd.to_datetime(bucket, utc=True)
        grp = ['MMSI','segment_id','bucket']
        dt_abs = (df['BaseDateTime'] - df['bucket']).abs()
        idx_min = dt_abs.groupby(df[grp].apply(tuple, axis=1)).idxmin()
        kept = df.loc[idx_min].copy().sort_values(['MMSI','segment_id','BaseDateTime'])
        self.audit['downsample'] = df.loc[~df.index.isin(kept.index)]
        return kept.drop(columns=['bucket'])

    def run_dataframe(self, df):
        df = self.enforce_domains(df)
        df = self.round_to_resolution(df)
        df = self.basic_hygiene(df)
        df = self.geofence_aoi(df)
        df = self.landmask_filter(df)
        df = self.sort_and_segment(df)
        df = self.kinematic_checks(df)
        df = self.jumps_and_spikes(df)
        df = self.turn_rate_sanity(df)
        df = self.clone_mmsi_resolution(df)
        df = self.downsample(df)
        return df.reset_index(drop=True).sort_values(['BaseDateTime'])

    def run(self, paths_or_df):
        df = self.load_and_normalize(paths_or_df)
        return self.run_dataframe(df), self.audit