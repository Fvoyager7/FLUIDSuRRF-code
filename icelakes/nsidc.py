import os
import gc
import re
import json
import shutil
import zipfile
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry.polygon import orient
from shapely.geometry import Polygon, mapping
from xml.etree import ElementTree as ET
from icelakes.utilities import get_size


def _make_earthdata_session(uid, pwd):
    """Authenticated requests session; trust_env=False avoids stale local proxy (e.g. 127.0.0.1:7897)."""
    session = requests.Session()
    session.trust_env = False
    session.auth = requests.auth.HTTPBasicAuth(uid, pwd)
    session.headers.update({'User-Agent': 'FLUIDSuRRF-icelakes/1.0 (requests)'})
    try:
        session.get('https://urs.earthdata.nasa.gov/home', timeout=30)
    except requests.RequestException:
        pass
    return session


def _print_earthdata_401_help():
    print('\nEarthdata login failed (401 Unauthorized). Please check:')
    print('  1. Username and password in ed/edcreds.py (log in at https://urs.earthdata.nasa.gov to verify).')
    print('  2. Authorize NSIDC data access for your account:')
    print('     https://urs.earthdata.nasa.gov/approve_app')
    print('     (approve applications related to NSIDC / ICESat-2).')
    print('  3. In a browser, open https://nsidc.org/data/atl03 and accept the data use agreement.\n')


def _download_url_earthdata(session, url, out_path, connect_timeout=30, read_timeout=7200, resume=True):
    """Stream download with Earthdata auth session; supports HTTP Range resume. Returns HTTP status code."""
    partial_bytes = 0
    if resume and os.path.exists(out_path):
        partial_bytes = os.path.getsize(out_path)
        if partial_bytes > 0:
            print('Resuming download from %s (%s already on disk)...' % (out_path, get_size(out_path)))

    headers = {}
    if partial_bytes > 0:
        headers['Range'] = 'bytes=%i-' % partial_bytes

    print('Connecting to Earthdata Cloud (first bytes may take 1–3 min)...')
    with session.get(url, stream=True, allow_redirects=True, headers=headers,
                     timeout=(connect_timeout, read_timeout)) as r:
        if r.status_code == 401:
            _print_earthdata_401_help()
            return 401
        if r.status_code == 416:
            # Range not satisfiable — file may already be complete
            return 200
        if r.status_code not in (200, 206):
            print('Download failed with HTTP status:', r.status_code)
            print('Final URL:', r.url)
            return r.status_code

        if r.status_code == 200 and partial_bytes > 0:
            print('Server did not honor resume; restarting download from scratch.')
            partial_bytes = 0

        content_length = int(r.headers.get('content-length', 0))
        if r.status_code == 206:
            total = partial_bytes + content_length
            downloaded = partial_bytes
            mode = 'ab'
        else:
            total = content_length
            downloaded = 0
            mode = 'wb'

        with open(out_path, mode) as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    print('  %.1f / %.1f MB' % (downloaded / 1e6, total / 1e6), end='\r', flush=True)
                else:
                    print('  %.1f MB' % (downloaded / 1e6), end='\r', flush=True)
    print()
    return 200


def _try_earthaccess_download(granule_id, granule_output_path, uid, pwd, data_url=None):
    """Login via earthaccess, then single-thread stream download (avoids pqdm hang on Windows)."""
    try:
        import earthaccess
        from earthaccess.exceptions import LoginAttemptFailure, LoginStrategyUnavailable
    except ImportError:
        print('Tip: pip install earthaccess may help Earthdata Cloud downloads in some network regions.')
        return None

    if not os.path.exists(granule_output_path):
        os.makedirs(granule_output_path)

    out_path = os.path.join(granule_output_path, granule_id)
    if os.path.exists(out_path):
        ok, msg = verify_local_granule(out_path, granule_id)
        if ok:
            print('Granule already downloaded: %s (%s)' % (out_path, get_size(out_path)))
            return out_path, 200
        print(msg)
        if os.path.getsize(out_path) > 0:
            print('Will attempt to resume partial download (HTTP Range)...')
        else:
            os.remove(out_path)

    if data_url is None:
        granule_search_url = 'https://cmr.earthdata.nasa.gov/search/granules'
        search_params = {
            'short_name': 'ATL03',
            'page_size': 1,
            'page_num': 1,
            'producer_granule_id': granule_id,
        }
        response = requests.get(granule_search_url, params=search_params,
                                headers={'Accept': 'application/json'}, timeout=60)
        entries = json.loads(response.content).get('feed', {}).get('entry', [])
        if not entries:
            print('No granule found in CMR for %s' % granule_id)
            return None
        links = entries[0].get('links', [])
        data_urls = [l['href'] for l in links
                     if str(l.get('rel', '')).endswith('/data#') and str(l.get('href', '')).startswith('http')]
        for url in data_urls:
            if 'earthdatacloud' in url:
                data_url = url
                break
        if data_url is None and data_urls:
            data_url = data_urls[0]
        if data_url is None:
            print('No HTTPS data link found in CMR metadata.')
            return None

    print('Downloading via earthaccess (single-thread stream, ~1.4 GB may take 10–60 min)...')
    print(' ', data_url)
    os.environ['EARTHDATA_USERNAME'] = uid
    os.environ['EARTHDATA_PASSWORD'] = pwd
    try:
        auth = earthaccess.login(strategy='environment', persist=False)
    except LoginAttemptFailure as e:
        print('earthaccess.login failed (wrong username/password?):')
        print(' ', str(e).split('\n')[0][:200])
        _print_earthdata_401_help()
        return None
    except LoginStrategyUnavailable as e:
        print('earthaccess.login failed:', e)
        return None

    if not auth.authenticated:
        print('earthaccess.login failed; check ed/edcreds.py.')
        _print_earthdata_401_help()
        return None

    # Pre-authorize NSIDC cloud URL (sets cookies for protected bucket)
    try:
        earthaccess.__store__.set_requests_session(data_url, method='head')
    except Exception as e:
        print('Warning: pre-authorize data URL failed:', e)

    session = earthaccess.get_requests_https_session()
    session.trust_env = False
    status = _download_url_earthdata(session, data_url, out_path,
                                     connect_timeout=30, read_timeout=7200, resume=True)
    if status != 200:
        return None

    ok, msg = verify_local_granule(out_path, granule_id)
    if not ok:
        print(msg)
        return None

    print('File to process: %s (%s)' % (out_path, get_size(out_path)))
    return out_path, 200


def get_cmr_granule_size_mb(granule_id):
    """Return CMR granule_size (MB) for a producer granule id."""
    granule_search_url = 'https://cmr.earthdata.nasa.gov/search/granules'
    search_params = {
        'short_name': 'ATL03',
        'page_size': 1,
        'page_num': 1,
        'producer_granule_id': granule_id,
    }
    response = requests.get(granule_search_url, params=search_params,
                            headers={'Accept': 'application/json'}, timeout=60)
    entries = json.loads(response.content).get('feed', {}).get('entry', [])
    if not entries:
        return None
    return float(entries[0]['granule_size'])


def verify_local_granule(filepath, granule_id):
    """Check local ATL03 .h5 exists, size matches CMR, and HDF5 opens."""
    if not os.path.isfile(filepath):
        return False, 'File not found: %s' % filepath

    local_size = os.path.getsize(filepath)
    expected_mb = get_cmr_granule_size_mb(granule_id)
    if expected_mb is not None:
        expected_bytes = int(expected_mb * 1024 * 1024)
        if local_size < expected_bytes * 0.98:
            return False, (
                'Incomplete download: %s is %s but CMR expects ~%.0f MB.\n'
                '  Delete this file and download again (browser or script).'
                % (filepath, get_size(filepath), expected_mb)
            )

    try:
        import h5py
        with h5py.File(filepath, 'r'):
            pass
    except OSError as e:
        return False, 'Corrupt/incomplete HDF5: %s\n  %s\n  Delete and re-download.' % (filepath, e)

    return True, 'OK'


def granule_id_to_cloud_url(granule_id):
    """Build Earthdata Cloud HTTPS URL from ATL03 granule filename (no CMR needed)."""
    m = re.match(r'ATL03_(\d{4})(\d{2})(\d{2}).*_(\d{3})_\d{2}\.h5', granule_id)
    if not m:
        return None
    y, mo, d, ver = m.groups()
    return ('https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected'
            '/ATLAS/ATL03/%s/%s/%s/%s/%s' % (ver, y, mo, d, granule_id))


##########################################################################################
    """
    Convert a shapefile to a geojson polygon file that can be used to 
    subset data from NSIDC. This already simplifies large polygons
    to reduce file size.

    Parameters
    ----------
    shapefile : string
        the path to the shapefile to convert
    output_directory : string
        the directory in which to write the geojson file

    Returns
    -------
    nothing

    Examples
    --------
    >>> shp2geojson_nsidc(my_shapefile.shp, output_directory = 'geojsons/')
    """    
    
    outfilename = shapefile.replace('.shp', '.geojson')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if outfilename[outfilename.rfind('/'):] != -1:
        outfilename = output_directory + outfilename[outfilename.rfind('/')+1:]
    else:
        outfilename = output_directory + outfilename

    gdf = gpd.read_file(shapefile)
    gdf.to_file(outfilename, driver='GeoJSON')
    print('Wrote file: %s' % outfilename)
    return outfilename
    
##########################################################################################    
def make_granule_list(geojson, start_date, end_date, icesheet, meltseason, list_out_name, geojson_dir_local='geojsons/', geojson_dir_remote=None, return_df=True, version=None):
    """
    Query for available granules over a region of interest and a start
    and end date. This will write a csv file with one column being the 
    available granule producer IDs and the other one the path to the 
    geojson file that needs to be used to subset them. 

    Parameters
    ----------
    geojson : string
        the filename of the geojson file
    start_date : string
        the start date in 'YYYY-MM-DD' format
    end_date : string
        the end date in 'YYYY-MM-DD' format
    list_out_name : string
        the path+filename of the csv file to be written out
    geojson_dir_local : string
        the path to the directory in which the geojson file is stored locally
    geojson_dir_remote : string
        the path to the directory in which the geojson file is stashed remotely
        if None (default) it will be the same as the local path

    Returns
    -------
    nothing, writes csv file to path given by list_out_name

    Examples
    --------
    >>> make_granule_list(my_geojson.geojson, '2021-05-01', '2021-09-15', 'auto', 
                          geojson_dir_local='geojsons/', geojson_dir_remote=None)
    """    

    short_name = 'ATL03'
    start_time = '00:00:00'
    end_time = '23:59:59'
    temporal = start_date + 'T' + start_time + 'Z' + ',' + end_date + 'T' + end_time + 'Z'

    cmr_collections_url = 'https://cmr.earthdata.nasa.gov/search/collections.json'
    granule_search_url = 'https://cmr.earthdata.nasa.gov/search/granules'
    base_url = 'https://n5eil02u.ecs.nsidc.org/egi/request'

    # Get json response from CMR collection metadata
    params = {'short_name': short_name}
    response = requests.get(cmr_collections_url, params=params)
    results = json.loads(response.content)

    # Find all instances of 'version_id' in metadata and print most recent version number
    if not version:
        versions = [el['version_id'] for el in results['feed']['entry']]
        latest_version = max(versions)
    else:
        latest_version = version
    capability_url = f'https://n5eil02u.ecs.nsidc.org/egi/capabilities/{short_name}.{latest_version}.xml'

    # read in geojson file
    gdf = gpd.read_file(geojson_dir_local + geojson)
    # poly = orient(gdf.simplify(0.05, preserve_topology=False).loc[0],sign=1.0)
    # polygon = ','.join([str(c) for xy in zip(*poly.exterior.coords.xy) for c in xy])
    polygon = ','.join([str(c) for xy in zip(*gdf.exterior.loc[0].coords.xy) for c in xy])
    search_params = {'short_name': short_name, 'version': latest_version, 'temporal': temporal, 'page_size': 100,
                     'page_num': 1,'polygon': polygon}

    # query for granules 
    granules = []
    headers={'Accept': 'application/json'}
    while True:
        response = requests.get(granule_search_url, params=search_params, headers=headers)
        results = json.loads(response.content)

        if len(results['feed']['entry']) == 0:
            break # Out of results, so break out of loop

        # Collect results and increment page_num
        granules.extend(results['feed']['entry'])
        search_params['page_num'] += 1

    granule_list, idx_unique = np.unique(np.array([g['producer_granule_id'] for g in granules]), return_index=True)
    granules = [g for i,g in enumerate(granules) if i in idx_unique]
    size_mb = [float(result["granule_size"]) for result in granules]
    
    print('Found %i %s version %s granules over %s between %s and %s.' % (len(granule_list), short_name, latest_version, 
                                                                          geojson, start_date, end_date))
    description = [icesheet + '_' + meltseason + '_' + geojson.replace('.geojson','')] * len(granule_list)
    if geojson_dir_remote is None:
        geojson_remote = geojson_dir_local + geojson
    else:
        geojson_remote = geojson_dir_remote + geojson

    thisdf = pd.DataFrame({'granule': granule_list, 
                           'geojson': geojson_remote, 
                           'description': description, 
                           'geojson_clip': geojson_remote.replace('simplified_', ''),
                           'size_mb': size_mb})
    if return_df:
        return thisdf
    else:
        if list_out_name == 'auto':
            list_out_name = 'granule_lists/' + geojson.replace('.geojson', ''), + '_' + start_date[:4] + '.csv'
        thisdf.to_csv(list_out_name, header=False, index=False)
        print('Wrote file: %s' % list_out_name)
    

##########################################################################################
def download_granule_cloud(granule_id, granule_output_path, uid, pwd):
    """
    Download a full ATL03 granule from Earthdata Cloud (CMR HTTPS link).
    Fallback when classic NSIDC EGI is unreachable (common SSL/network issues).
    Spatial subsetting is applied later locally by detect_lakes.
    """
    granule_search_url = 'https://cmr.earthdata.nasa.gov/search/granules'
    search_params = {
        'short_name': 'ATL03',
        'page_size': 1,
        'page_num': 1,
        'producer_granule_id': granule_id,
    }
    data_url = None
    try:
        response = requests.get(granule_search_url, params=search_params,
                                headers={'Accept': 'application/json'}, timeout=60)
        entries = json.loads(response.content).get('feed', {}).get('entry', [])
        if entries:
            links = entries[0].get('links', [])
            data_urls = [l['href'] for l in links if str(l.get('rel', '')).endswith('/data#') and str(l.get('href', '')).startswith('http')]
            for url in data_urls:
                if 'earthdatacloud' in url:
                    data_url = url
                    break
            if data_url is None and data_urls:
                data_url = data_urls[0]
    except requests.RequestException as e:
        print('CMR lookup failed (%s); using constructed cloud URL.' % type(e).__name__)

    if data_url is None:
        data_url = granule_id_to_cloud_url(granule_id)
    if data_url is None:
        print('No HTTPS data link found for cloud download.')
        return 'none', 404

    if not os.path.exists(granule_output_path):
        os.makedirs(granule_output_path)

    print('Downloading full granule from Earthdata Cloud (no server-side subset):')
    print(' ', data_url)

    alt = _try_earthaccess_download(granule_id, granule_output_path, uid, pwd, data_url=data_url)
    if alt is not None:
        return alt

    return 'none', 401


##########################################################################################
# @profile
def download_granule(granule_id, gtxs, geojson, granule_output_path, uid, pwd, vars_sub='default', spatial_sub=False):
    """
    Download a single ICESat-2 ATL03 granule based on its producer ID,
    subsets it to a given geojson file, and puts it into the specified
    output directory as a .h5 file. A NASA earthdata user id (uid), and
    the associated password are required. 
    (Can also provide a shapefile instead of geojson.)

    Parameters
    ----------
    granule_id : string
        the producer_granule_id for CMR search
    gtxs : string or list
        the ground tracks to request
        possible values:
            'gt1l' or 'gt1r' or 'gt2l', ... (single gtx)
            ['gt1l', 'gt3r', ...] (list of gtxs)
    geojson : string
        filepath to the geojson file used for spatial subsetting
    granule_output_path : string
        folder in which to save the subsetted granule
    uid : string
        earthdata user id
    pwd : string
        the associated password

    Returns
    -------
    nothing

    Examples
    --------
    >>> download_granule_nsidc(granule_id='ATL03_20210715182907_03381203_005_01.h5', 
                               geojson='geojsons/jakobshavn.geojson', 
                               gtxs='all'
                               granule_output_path='IS2data', 
                               uid='myuserid', 
                               pwd='mypasword')
    """
    print('--> parameters: granule_id = %s' % granule_id)
    print('                gtxs = %s' % gtxs)
    print('                geojson = %s' % geojson)
    print('                granule_output_path = %s' % granule_output_path)
    print('                vars_sub = %s' % vars_sub)
    print('                spatial_sub = %s\n' % spatial_sub)
    
    short_name = 'ATL03'
    version = granule_id[30:33]
    granule_search_url = 'https://cmr.earthdata.nasa.gov/search/granules'
    capability_url = f'https://n5eil02u.ecs.nsidc.org/egi/capabilities/{short_name}.{version}.xml'
    base_url = 'https://n5eil02u.ecs.nsidc.org/egi/request'
    
    geojson_filepath = str(os.getcwd() + '/' + geojson)
    
    # set the variables for subsetting
    if vars_sub == 'default':
        vars_sub = ['/ancillary_data/atlas_sdp_gps_epoch',
                    '/ancillary_data/calibrations/dead_time/gtx',
                    '/orbit_info/rgt',
                    '/orbit_info/cycle_number',
                    '/orbit_info/sc_orient',
                    '/gtx/geolocation/segment_id',
                    '/gtx/geolocation/ph_index_beg',
                    '/gtx/geolocation/segment_dist_x',
                    '/gtx/geolocation/segment_length',
                    '/gtx/geolocation/segment_ph_cnt',
                    # '/gtx/geophys_corr/dem_h',
                    '/gtx/geophys_corr/geoid',
                    '/gtx/bckgrd_atlas/pce_mframe_cnt',
                    '/gtx/bckgrd_atlas/tlm_height_band1',
                    '/gtx/bckgrd_atlas/tlm_height_band2',
                    '/gtx/bckgrd_atlas/tlm_top_band1',
                    '/gtx/bckgrd_atlas/tlm_top_band2',
                    # '/gtx/bckgrd_atlas/bckgrd_counts',
                    # '/gtx/bckgrd_atlas/bckgrd_int_height',
                    # '/gtx/bckgrd_atlas/delta_time',
                    '/gtx/heights/lat_ph',
                    '/gtx/heights/lon_ph',
                    '/gtx/heights/h_ph',
                    '/gtx/heights/delta_time',
                    '/gtx/heights/dist_ph_along',
                    '/gtx/heights/quality_ph',
                    # '/gtx/heights/signal_conf_ph',
                    '/gtx/heights/pce_mframe_cnt',
                    '/gtx/heights/ph_id_pulse'
                    ]
        if int(version) > 5:
            vars_sub.append('/gtx/heights/weight_ph')
    beam_list = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
    
    if gtxs == 'all':
        var_list = sum([[v.replace('/gtx','/'+bm) for bm in beam_list] if '/gtx' in v else [v] for v in vars_sub],[])
    elif type(gtxs) == str:
        var_list = [v.replace('/gtx','/'+gtxs.lower()) if '/gtx' in v else v for v in vars_sub]
    elif type(gtxs) == list:
        var_list = sum([[v.replace('/gtx','/'+bm.lower()) for bm in gtxs] if '/gtx' in v else [v] for v in vars_sub],[])
    else: # default to requesting all beams
        var_list = sum([[v.replace('/gtx','/'+bm) for bm in beam_list] if '/gtx' in v else [v] for v in vars_sub],[])
    
    # search for the given granule
    search_params = {
        'short_name': short_name,
        'page_size': 100,
        'page_num': 1,
        'producer_granule_id': granule_id}

    granules = []
    headers={'Accept': 'application/json'}
    while True:
        response = requests.get(granule_search_url, params=search_params, headers=headers)
        results = json.loads(response.content)

        if len(results['feed']['entry']) == 0:
            # Out of results, so break out of loop
            break

        # Collect results and increment page_num
        granules.extend(results['feed']['entry'])
        search_params['page_num'] += 1
        
    granule_list, idx_unique = np.unique(np.array([g['producer_granule_id'] for g in granules]), return_index=True)
    granules = [g for i,g in enumerate(granules) if i in idx_unique] # keeps double counting, not sure why
    print('\nDownloading ICESat-2 data. Found granules:')
    if len(granules) == 0:
        print('None')
        return 'none', 404
    for result in granules:
        print('  '+result['producer_granule_id'], f', {float(result["granule_size"]):.2f} MB',sep='')
        
    # Use geopandas to read in polygon file as GeoDataFrame object 
    # Note: a shapefile, KML, or almost any other vector-based spatial data format could be substituted here.
    gdf = gpd.read_file(geojson_filepath)

    # make sure the two regions that go over the date line are adjusted 
    # if ('West_Ep-F.geojson' in geojson_filepath) or ('East_E-Ep.geojson' in geojson_filepath): 
    #     lon180 = np.array(gdf.geometry.iloc[0].exterior.coords.xy[0])
    #     lon180[lon180 < 0] = lon180[lon180 < 0]  + 360
    #     gdf['geometry'] = Polygon(list(zip(lon180, gdf.geometry.iloc[0].exterior.coords.xy[1])))
    #     poly = orient(gdf.loc[0].geometry,sign=1.0)
    #     lon180 = np.array(poly.exterior.coords.xy[0])
    #     # lon180[lon180 >= 180] = lon180[lon180 >= 180] - 360
    #     gdf['geometry'] = Polygon(list(zip(lon180, gdf.geometry.iloc[0].exterior.coords.xy[1])))
    #     poly = gdf.loc[0].geometry
    
    # Simplify polygon for complex shapes in order to pass a reasonable request length to CMR. 
    # The larger the tolerance value, the more simplified the polygon.
    # Orient counter-clockwise: CMR polygon points need to be provided in counter-clockwise order. 
    # The last point should match the first point to close the polygon.
    # poly = orient(gdf.simplify(0.05, preserve_topology=False).loc[0],sign=1.0)
    # else:
    poly = orient(gdf.loc[0].geometry,sign=1.0)

    geojson_data = gpd.GeoSeries(poly).to_json() # Convert to geojson
    geojson_data = geojson_data.replace(' ', '') #remove spaces for API call
    
    #Format dictionary to polygon coordinate pairs for CMR polygon filtering
    polygon = ','.join([str(c) for xy in zip(*poly.exterior.coords.xy) for c in xy])
    
    print('\nInput geojson:', geojson)
    print('Simplified polygon coordinates based on geojson input:', polygon)
    
    # Create session to store cookie and pass credentials to capabilities url
    session = requests.session()
    session.trust_env = False
    try:
        s = session.get(capability_url, timeout=15)
        response = session.get(s.url, auth=(uid, pwd), timeout=15)
    except (requests.exceptions.SSLError, requests.exceptions.ConnectionError,
            requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout) as e:
        print('\nNSIDC EGI endpoint unreachable (%s).' % type(e).__name__)
        print('Falling back to Earthdata Cloud full-granule download.')
        print('Spatial subsetting will be applied locally after download.\n')
        return download_granule_cloud(granule_id, granule_output_path, uid, pwd)

    try:
        root = ET.fromstring(response.content)
    except:
        try:
            cont = str(request._content)
            print('request status code:', request.status_code)
            the_code = cont[cont.find('<Code>')+6:cont.find('</Code>')]
            if len(the_code) < 1000:
                print(the_code)
            the_message = cont[cont.find('<Message>')+9:cont.find('</Message>')]
            if len(the_message) < 5000:
                print(the_message)
            print('')
            return 'none', response.status_code
        except:
            return 'none', response.status_code

    #collect lists with each service option
    subagent = [subset_agent.attrib for subset_agent in root.iter('SubsetAgent')]
    
    # this is for getting possible variable values from the granule search
    if len(subagent) > 0 :
        # variable subsetting
        variables = [SubsetVariable.attrib for SubsetVariable in root.iter('SubsetVariable')]  
        variables_raw = [variables[i]['value'] for i in range(len(variables))]
        variables_join = [''.join(('/',v)) if v.startswith('/') == False else v for v in variables_raw] 
        variable_vals = [v.replace(':', '/') for v in variables_join]
    
    # make sure to only request the variables that are available
    def intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3
    if vars_sub == 'all':
        var_list_subsetting = ''
    else:
        var_list_subsetting = intersection(variable_vals,var_list)
    
    if len(subagent) < 1 :
        print('No services exist for', short_name, 'version', latest_version)
        agent = 'NO'
        coverage,Boundingshape,polygon = '','',''
    else:
        agent = ''
        subdict = subagent[0]
        if (subdict['spatialSubsettingShapefile'] == 'true') and spatial_sub:
            ######################################## Boundingshape = geojson_data
            Boundingshape = polygon
        else:
            Boundingshape, polygon = '',''
        coverage = ','.join(var_list_subsetting)
    if (vars_sub=='all') & (not spatial_sub):
        agent = 'NO'
        
    page_size = 100
    request_mode = 'stream'
    page_num = int(np.ceil(len(granules)/page_size))

    param_dict = {'short_name': short_name, 
                  'producer_granule_id': granule_id,
                  'version': version,  
                  'polygon': polygon,
                  'Boundingshape': Boundingshape,  
                  'Coverage': coverage, 
                  'page_size': page_size, 
                  'request_mode': request_mode, 
                  'agent': agent, 
                  'email': 'yes'}

    #Remove blank key-value-pairs
    param_dict = {k: v for k, v in param_dict.items() if v != ''}

    #Convert to string
    param_string = '&'.join("{!s}={!r}".format(k,v) for (k,v) in param_dict.items())
    param_string = param_string.replace("'","")

    #Print API base URL + request parameters
    endpoint_list = [] 
    for i in range(page_num):
        page_val = i + 1
        API_request = api_request = f'{base_url}?{param_string}&page_num={page_val}'
        endpoint_list.append(API_request)

    print('\nAPI request URL:')
    print(*endpoint_list, sep = "\n") 
    
    # Create an output folder if the folder does not already exist.
    path = str(os.getcwd() + '/' + granule_output_path)
    if not os.path.exists(path):
        os.mkdir(path)

    # Different access methods depending on request mode:
    for i in range(page_num):
        page_val = i + 1
        print('\nOrder: ', page_val)
        print('Requesting...')
        request = session.get(base_url, params=param_dict)
        print('HTTP response from order response URL: ', request.status_code)
        # try: 
        #     cont = str(request._content)
        #     print(cont[cont.find('<Code>')+6:cont.find('</Code>')],
        #       '(', cont[cont.find('<Message>')+9:cont.find('</Message>')], ')\n')
        # except:
        #     pass
        request.raise_for_status()
        d = request.headers['content-disposition']
        fname = re.findall('filename=(.+)', d)
        dirname = os.path.join(path,fname[0].strip('\"'))
        print('Downloading...')
        open(dirname, 'wb').write(request.content)
        print('Data request', page_val, 'is complete.')

    # Unzip outputs
    for z in os.listdir(path): 
        if z.endswith('.zip'): 
            zip_name = path + "/" + z 
            zip_ref = zipfile.ZipFile(zip_name) 
            zip_ref.extractall(path) 
            zip_ref.close() 
            os.remove(zip_name) 

    # Clean up Outputs folder by removing individual granule folders 
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            try:
                shutil.move(os.path.join(root, file), path)
            except OSError:
                pass
        for name in dirs:
            os.rmdir(os.path.join(root, name))
            
    print('\nUnzipped files and cleaned up directory.')
    print('Output data saved in:', granule_output_path)
    
    filelist = [granule_output_path+'/'+f for f in os.listdir(granule_output_path) \
                if os.path.isfile(os.path.join(granule_output_path, f)) & (granule_id in f)]
    
    if len(filelist) == 0: 
        return 'none'
    else:
        filename = filelist[0]
    print('File to process: %s (%s)' % (filename, get_size(filename)))
    
    print(filename, 'status:', request.status_code)
    return filename, request.status_code


##########################################################################################
def print_granule_stats(photon_data, bckgrd_data, ancillary, outfile=None):
    """
    Print stats from a read-in granule.
    Mostly for checking that things are working / OSG testing. 

    Parameters
    ----------
    photon_data : dict of pandas dataframes
        the first output of read_atl03()
    bckgrd_data : dict of pandas dataframes
        the second output of read_atl03()
    ancillary : dict
        the third output of read_atl03()
    outfile : string
        file path and name for the output
        if outfile=None, results are printed to stdout

    Returns
    -------
    nothing
                                    
    Examples
    --------
    >>> print_granule_stats(photon_data, bckgrd_data, ancillary, outfile='stats.txt')
    """    

    if outfile is not None: 
        import sys
        original_stdout = sys.stdout
        f = open(outfile, "w")
        sys.stdout = f

    print('\n*********************************')
    print('** GRANULE INFO AND STATISTICS **')
    print('*********************************\n')
    print(ancillary['granule_id'])
    print('RGT:', ancillary['rgt'])
    print('cycle number:', ancillary['cycle_number'])
    print('spacecraft orientation:', ancillary['sc_orient'])
    print('beam configuation:')
    for k in ancillary['gtx_beam_dict'].keys():
        print(' ', k, ': beam', ancillary['gtx_beam_dict'][k], '(%s)'%ancillary['gtx_strength_dict'][k])
    for k in photon_data.keys():
        counts = photon_data[k].count()
        nanvals = photon_data[k].isna().sum()
        maxs = photon_data[k].max()
        mins = photon_data[k].min()
        print('\nPHOTON DATA SUMMARY FOR BEAM %s'%k.upper())
        print(pd.DataFrame({'count':counts, 'nans':nanvals, 'min':mins, 'max':maxs}))
        
    if outfile is not None:
        f.close()
        sys.stdout = original_stdout
        with open(outfile, 'r') as f:
            print(f.read())
    return


##########################################################################################
class edc:
    # Legacy encrypted-credential placeholders (used by detect_lakes.py download path).
    # Listing granules via CMR does not need these. Prefer ed/edcreds.py for local runs.
    u = b"<paste your encrypted user id here>"
    p = b'<paste your encrypted password here>'