#!/usr/bin/env python3
"""Descarga chips Sentinel-2 pre/post y derivados rNBR desde Google Earth Engine.
Cada evento se exporta como un chip 512x512 a 10m (5.12 x 5.12 km), centrado en el
centro del evento agrupado. No crea labels finales: genera label_seed con 0/1/255.
255 = incertidumbre/no-data; esas zonas no deben usarse como ground truth final.
"""
import argparse,csv,io,json,os,sys,time,zipfile
from datetime import datetime,timedelta
from pathlib import Path
import requests
import ee

BANDS=['B2','B3','B4','B8','B11','B12']
SCALE=10
CRS='EPSG:32721'  # UTM 21S, Santa Fe
HALF_SIDE_M=2560 # 512 pixeles x 10m / 2
PRE_START_DAYS=75
PRE_END_DAYS=15
POST_START_DAYS=5
POST_END_DAYS=50


def mask_s2(img):
    scl=img.select('SCL')
    # descarta no-data, sombras, nubes, cirrus, nieve; mantiene vegetacion/suelo/agua
    good=scl.neq(0).And(scl.neq(1)).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return img.updateMask(good).select(BANDS).multiply(0.0001).copyProperties(img,['system:time_start'])

def composite(roi,start,end):
    c=(ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
       .filterBounds(roi).filterDate(start,end).filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE',80)).map(mask_s2))
    n=c.size().getInfo()
    if n==0: raise RuntimeError(f'Sin Sentinel-2 para {start} a {end}')
    return c.median().clip(roi),n

def idx(img):
    nbr=img.normalizedDifference(['B8','B12']).rename('NBR')
    ndvi=img.normalizedDifference(['B8','B4']).rename('NDVI')
    ndmi=img.normalizedDifference(['B8','B11']).rename('NDMI')
    mndwi=img.normalizedDifference(['B3','B11']).rename('MNDWI')
    return nbr,ndvi,ndmi,mndwi

def get_zip(image, roi, name, out_tif):
    params = {
        'name': name,
        'region': roi,
        'scale': SCALE,
        'crs': CRS,
        'format': 'GEO_TIFF',
        'filePerBand': False
    }

    url = image.getDownloadURL(params)
    r = requests.get(url, timeout=600)
    r.raise_for_status()

    out_tif.parent.mkdir(parents=True, exist_ok=True)

    # Earth Engine puede devolver ZIP o GeoTIFF directo.
    if r.content[:2] == b'PK':
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            tifs = [
                n for n in z.namelist()
                if n.lower().endswith(('.tif', '.tiff'))
            ]

            if not tifs:
                raise RuntimeError(
                    f'No se encontró GeoTIFF dentro del ZIP: {z.namelist()}'
                )

            with z.open(tifs[0]) as src, open(out_tif, 'wb') as dst:
                dst.write(src.read())

    # GeoTIFF clásico: bytes II* o MM*
    elif r.content[:4] in (b'II*\x00', b'MM\x00*'):
        with open(out_tif, 'wb') as dst:
            dst.write(r.content)

    else:
        preview = r.content[:300].decode('utf-8', errors='replace')
        raise RuntimeError(
            f'Respuesta inesperada de Earth Engine. Inicio de respuesta: {preview}'
        )


def process_event(row,root,overwrite=False):
    eid=row['event_id']; d=datetime.fromisoformat(row['event_date'])
    point=ee.Geometry.Point([float(row['lon']),float(row['lat'])])
    roi=point.transform(CRS,1).buffer(HALF_SIDE_M,1).bounds(1,CRS)
    pre_s=(d-timedelta(days=PRE_START_DAYS)).date().isoformat(); pre_e=(d-timedelta(days=PRE_END_DAYS)).date().isoformat()
    post_s=(d+timedelta(days=POST_START_DAYS)).date().isoformat(); post_e=(d+timedelta(days=POST_END_DAYS)).date().isoformat()
    pre,npre=composite(roi,pre_s,pre_e);post,npost=composite(roi,post_s,post_e)
    nbr_pre,ndvi_pre,ndmi_pre,mndwi_pre=idx(pre);nbr_post,_,_,_=idx(post)
    dnbr=nbr_pre.subtract(nbr_post).rename('dNBR')
    rnbr=dnbr.divide(nbr_pre.abs().sqrt().max(ee.Image.constant(0.1))).rename('rNBR')
    # Seed: 1 burn candidate, 0 non-burn candidate, 255 uncertain/no data.
    valid=pre.mask().reduce(ee.Reducer.min()).And(post.mask().reduce(ee.Reducer.min()))
    burn=rnbr.gte(0.10).And(ndvi_pre.gte(0.10))
    nonburn=rnbr.lte(0.05)
    seed=ee.Image(255).where(nonburn,0).where(burn,1).updateMask(valid).rename('label_seed').toUint8()
    aux=(nbr_pre.rename('NBR_pre').addBands(nbr_post.rename('NBR_post')).addBands(dnbr).addBands(rnbr)
         .addBands(ndvi_pre).addBands(ndmi_pre).addBands(mndwi_pre).addBands(valid.rename('valid')).clip(roi))
    outputs=[
      (pre.select(BANDS),root/'data/raw/pre'/f'{eid}_pre.tif'),
      (post.select(BANDS),root/'data/raw/post'/f'{eid}_post.tif'),
      (aux,root/'data/raw/aux'/f'{eid}_aux.tif'),
      (seed,root/'data/raw/labels'/f'{eid}_label_seed.tif'),
    ]
    for image,path in outputs:
        if overwrite or not path.exists(): get_zip(image,roi,path.stem,path)
    return {'event_id':eid,'pre_start':pre_s,'pre_end':pre_e,'post_start':post_s,'post_end':post_e,'n_pre':npre,'n_post':npost,'status':'ok'}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--events',default='events/events_santafe_dedup.csv')
    ap.add_argument('--root',default='.')
    ap.add_argument('--project',required=True,help='Google Cloud project registrado para Earth Engine')
    ap.add_argument('--limit',type=int,default=10,help='Cantidad de eventos a descargar; use 0 para todos')
    ap.add_argument('--start',type=int,default=0)
    ap.add_argument('--overwrite',action='store_true')
    args=ap.parse_args();root=Path(args.root).resolve()
    ee.Initialize(project=args.project)
    with open(args.events,encoding='utf-8') as f: rows=list(csv.DictReader(f))
    rows=rows[args.start:] if args.limit==0 else rows[args.start:args.start+args.limit]
    manifest=[]
    for k,row in enumerate(rows,1):
        print(f'[{k}/{len(rows)}] {row["event_id"]} {row["event_date"]}')
        try:
            rec=process_event(row,root,args.overwrite);print('  OK',rec['n_pre'],rec['n_post'])
        except Exception as e:
            rec={'event_id':row['event_id'],'status':'error','error':str(e)};print('  ERROR',e,file=sys.stderr)
        manifest.append(rec)
    mp=root/'data/manifests/download_manifest.json'
    mp.parent.mkdir(parents=True,exist_ok=True)
    existing=[]
    if mp.exists():
        try: existing=json.loads(mp.read_text())
        except Exception: pass
    mp.write_text(json.dumps(existing+manifest,indent=2))
    print('Manifest:',mp)
if __name__=='__main__':main()
