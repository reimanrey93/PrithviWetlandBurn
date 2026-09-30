#!/usr/bin/env python3
"""Agrupa focos repetidos en eventos espacio-temporales para evitar duplicados.
Regla por defecto: puntos a <= 2 km y <= 14 días pertenecen al mismo evento.
Salida: events_santafe_dedup.csv. Los splits posteriores deben hacerse por event_id.
"""
import argparse, csv, math
from datetime import datetime
from pathlib import Path

R=6371.0088

def hav_km(a,b):
    lon1,lat1=map(math.radians,(a['lon'],a['lat']))
    lon2,lat2=map(math.radians,(b['lon'],b['lat']))
    dlon, dlat=lon2-lon1,lat2-lat1
    h=math.sin(dlat/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(h))

class UF:
    def __init__(self,n): self.p=list(range(n))
    def find(self,x):
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]];x=self.p[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.p[b]=a

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='events/events_raw_santafe.csv')
    ap.add_argument('--output',default='events/events_santafe_dedup.csv')
    ap.add_argument('--distance-km',type=float,default=2.0)
    ap.add_argument('--days',type=int,default=14)
    args=ap.parse_args()
    rows=[]
    with open(args.input,encoding='utf-8') as f:
        for r in csv.DictReader(f):
            r['lon']=float(r['lon']);r['lat']=float(r['lat']);r['dt']=datetime.fromisoformat(r['date'])
            rows.append(r)
    uf=UF(len(rows))
    for i in range(len(rows)):
        for j in range(i+1,len(rows)):
            if abs((rows[j]['dt']-rows[i]['dt']).days)<=args.days and hav_km(rows[i],rows[j])<=args.distance_km:
                uf.union(i,j)
    groups={}
    for i,r in enumerate(rows):groups.setdefault(uf.find(i),[]).append(r)
    out=[]
    for n,grp in enumerate(sorted(groups.values(),key=lambda g:min(x['dt'] for x in g)),start=1):
        # Centro mediano para ser robusto a focos múltiples del mismo evento.
        lon=sorted(x['lon'] for x in grp)[len(grp)//2]
        lat=sorted(x['lat'] for x in grp)[len(grp)//2]
        start=min(x['dt'] for x in grp).date().isoformat(); end=max(x['dt'] for x in grp).date().isoformat()
        out.append({'event_id':f'SF_{n:03d}','lon':f'{lon:.6f}','lat':f'{lat:.6f}','event_date':start,'last_detection':end,'n_points':len(grp),'source_ids':'|'.join(x['source_id'] for x in grp)})
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    with open(args.output,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0].keys()));w.writeheader();w.writerows(out)
    print(f'Eventos originales: {len(rows)}')
    print(f'Eventos agrupados: {len(out)}')
    print(f'Archivo: {args.output}')
if __name__=='__main__':main()
