"""Segmenting the Smartwatch Market - full cluster analysis (run: python smartwatch_analysis.py)
Requires pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, openpyxl. Writes charts to ./out/"""
from builtins import print
import os; os.makedirs("out",exist_ok=True)

# ======================================================================
# STEP 0 - DATA PREP
# ======================================================================
import pandas as pd, numpy as np
f=r"C:\Users\sykob\Downloads\Smartwatch Data.xlsx"
seg=pd.read_excel(f,sheet_name='Segmentation Data',header=3).iloc[:,1:]
dis=pd.read_excel(f,sheet_name='Discrimination Data',header=3).iloc[:,1:]
seg=seg.rename(columns={seg.columns[0]:'ID'}); dis=dis.rename(columns={dis.columns[0]:'ID'})
print(seg.shape,dis.shape); print(seg.dtypes.to_string()); print(dis.dtypes.to_string())
print('missing seg',seg.isna().sum().sum(),'missing dis',dis.isna().sum().sum())
print('dup IDs',seg.ID.duplicated().sum(), 'ID match',(seg.ID.values==dis.ID.values).all())
print('dup rows (excl ID) seg',seg.drop(columns='ID').duplicated().sum(),' combined',pd.concat([seg,dis.drop(columns='ID')],axis=1).drop(columns='ID').duplicated().sum())
print(seg.describe().T.round(2).to_string())
print(dis.describe().T.round(2).to_string())
rat=seg.columns[1:13]
print('rating range',seg[rat].min().min(),seg[rat].max().max())
for c in ['Price','Age']:
    s=(seg if c=='Price' else dis)[c]
    q1,q3=s.quantile([.25,.75]); i=q3-q1
    print(c,'IQR fence',q1-1.5*i,q3+1.5*i,'n out',((s<q1-1.5*i)|(s>q3+1.5*i)).sum(),'z>3',(abs((s-s.mean())/s.std())>3).sum(), 'min',s.min(),'max',s.max())
print(seg.Price.sort_values().head(5).tolist(), seg.Price.sort_values().tail(8).tolist())
print(dis.Age.sort_values().head(5).tolist(), dis.Age.sort_values().tail(8).tolist())
work=['Health','Finc','Sales','Advt','Edu','Cons','Eng','Tech','Retail','SMB']
print('work rows sum', dis[work].sum(axis=1).value_counts().to_dict())
print(dis[['iPhone','CompBuy','Female','Degree','Income']].apply(lambda s:s.value_counts().to_dict()).to_string())


# ======================================================================
# Q1 - HOW MANY SEGMENTS
# ======================================================================
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
rat=['Innov','ConstCom','CreatCom','TimelyInf','SaveM_T','SaveM_L','TaskMgm','DeviceSt','Photo','Wellness','Athlete','Style']
X=seg[rat].values
print('SDs:',seg[rat].std().round(2).to_dict())
print('Ratio max/min SD',round(seg[rat].std().max()/seg[rat].std().min(),2))
Z=StandardScaler().fit_transform(X)
# Ward
L=linkage(Z,'ward')
fig,ax=plt.subplots(figsize=(11,5))
dendrogram(L,truncate_mode='lastp',p=30,ax=ax,show_leaf_counts=True,color_threshold=0.0,above_threshold_color='#555')
ax.set_title("Ward's hierarchical clustering dendrogram (last 30 merges)"); ax.set_ylabel('Ward distance')
for k in (3,4):
    h=(L[-k+1,2]+L[-k,2])/2; ax.axhline(h,ls='--',c='C%d'%k,label=f'cut for k={k}')
ax.legend(); plt.tight_layout(); plt.savefig('out/dendrogram.png',dpi=130); plt.close()
print('last merge heights (k=10..2):',[round(L[-i,2],1) for i in range(9,0,-1)])
print('height jumps going to k clusters:',{k:round(L[-(k-1),2]-L[-k,2],1) for k in range(2,9)})
rows=[]; labs={}
for k in range(2,9):
    km=KMeans(k,n_init=50,random_state=42).fit(Z); l=km.labels_; labs[k]=l
    w=fcluster(L,k,'maxclust')
    sz=np.bincount(l); 
    rows.append(dict(k=k,inertia=km.inertia_,silhouette=silhouette_score(Z,l),CH=calinski_harabasz_score(Z,l),DB=davies_bouldin_score(Z,l),min_n=sz.min(),min_pct=100*sz.min()/1000,max_n=sz.max(),ARI_ward_vs_km=adjusted_rand_score(w,l)))
res=pd.DataFrame(rows); print(res.round(3).to_string(index=False))
# stability: bootstrap ARI (fit on bootstrap, predict full, compare to full-sample solution)
rng=np.random.default_rng(1); stab={}
for k in range(2,7):
    base=labs[k]; a=[]
    for b in range(100):
        idx=rng.integers(0,1000,1000)
        m=KMeans(k,n_init=10,random_state=b).fit(Z[idx]); a.append(adjusted_rand_score(base,m.predict(Z)))
    stab[k]=(np.mean(a),np.min(a))
print('bootstrap stability (mean ARI, min ARI):',{k:(round(v[0],3),round(v[1],3)) for k,v in stab.items()})
# split-half
sh={}
for k in range(2,7):
    idx=rng.permutation(1000); A,B=idx[:500],idx[500:]
    mA=KMeans(k,n_init=20,random_state=0).fit(Z[A]); mB=KMeans(k,n_init=20,random_state=0).fit(Z[B])
    sh[k]=adjusted_rand_score(mA.predict(Z[B]),mB.labels_)
print('split-half ARI:',{k:round(v,3) for k,v in sh.items()})
fig,ax=plt.subplots(1,3,figsize=(14,4))
ks=res.k
ax[0].plot(ks,res.inertia,'o-'); ax[0].set_title('Elbow (inertia / within-cluster SS)'); 
ax[1].plot(ks,res.silhouette,'o-',c='C1'); ax[1].set_title('Mean silhouette')
ax[2].plot(ks,res.CH,'o-',c='C2'); ax[2].set_title('Calinski-Harabasz')
for a in ax: a.set_xlabel('k'); a.axvline(3,ls=':',c='grey'); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig('out/k_selection.png',dpi=130); plt.close()
res.to_csv('out/k_selection.csv',index=False)
np.save('out/Z.npy',Z)
for k in (3,4,5):
    print(k,'sizes',np.bincount(labs[k]).tolist())


# ======================================================================
# Q2 - DESCRIBE AND NAME SEGMENTS
# ======================================================================
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
rat=['Innov','ConstCom','CreatCom','TimelyInf','SaveM_T','SaveM_L','TaskMgm','DeviceSt','Photo','Wellness','Athlete','Style']
Z=np.load('out/Z.npy')
km=KMeans(3,n_init=50,random_state=42).fit(Z); lab=km.labels_
df=pd.concat([seg.drop(columns='ID'),dis.drop(columns='ID')],axis=1)
df['Other']=(df[['Health','Finc','Sales','Advt','Edu','Cons','Eng','Tech','Retail','SMB']].sum(axis=1)==0).astype(int)
# order segments by label via nearest to reference: name later; first show raw means
df['seg']=lab
means=df.groupby('seg')[rat].mean()
# cross-check vs reference workbook (original-scale means)
ref=pd.read_excel(r"C:\Users\sykob\Downloads\smartwatch_cluster_analysis.xlsx",sheet_name='Original Scale Means')
refm=ref.iloc[:,4:].values; refn=ref['Cluster name'].tolist()
from scipy.optimize import linear_sum_assignment
D=np.linalg.norm(means.values[:,None,:]-refm[None,:,:],axis=2); r,c=linear_sum_assignment(D)
print('Match mine->ref:',{int(a):refn[b] for a,b in zip(r,c)},'max centroid diff (rating pts):',round(np.abs(means.values[r]-refm[c]).max(),3))
print('sizes mine',np.bincount(lab).tolist(),'ref',ref['N'].tolist())
nm={int(a):refn[b] for a,b in zip(r,c)}
# order: by ref order
order=[a for b,a in sorted((b,a) for a,b in zip(r,c))]
remap={old:new+1 for new,old in enumerate(order)}
df['seg']=df['seg'].map(remap); means=df.groupby('seg')[rat].mean()
df.to_pickle('out/df.pkl')
n=df.seg.value_counts().sort_index(); print((n/10).round(1).to_dict(), n.to_dict())
tab=means.copy(); tab.loc['Overall']=df[rat].mean(); print(tab.round(2).T.to_string())
print('diff vs overall');print((means-df[rat].mean()).round(2).T.to_string())
# ANOVA
an=[]
for v in rat:
    g=[df.loc[df.seg==s,v] for s in (1,2,3)]; F,p=stats.f_oneway(*g)
    ssb=sum(len(x)*(x.mean()-df[v].mean())**2 for x in g); sst=((df[v]-df[v].mean())**2).sum()
    kw=stats.kruskal(*g)[1]
    an.append((v,F,p,min(p*12,1),ssb/sst,kw))
an=pd.DataFrame(an,columns=['var','F','p','p_bonf','eta2','KW_p']).sort_values('F',ascending=False); print(an.round(4).to_string(index=False))
# heatmap
fig,ax=plt.subplots(1,2,figsize=(13,5.5))
import seaborn as sns
sns.heatmap(means.T,annot=True,fmt='.2f',cmap='viridis',ax=ax[0],cbar_kws={'label':'mean importance (1-7)'})
ax[0].set_title('Mean importance by segment'); 
sns.heatmap((means-df[rat].mean()).T,annot=True,fmt='+.2f',cmap='RdBu_r',center=0,ax=ax[1],cbar_kws={'label':'diff vs overall mean'})
ax[1].set_title('Deviation from overall mean (red=high, blue=low)')
plt.tight_layout(); plt.savefig('out/heatmap.png',dpi=130); plt.close()
# descriptors
cont=['Price','Age','Income']; 
print('\nCONTINUOUS')
for v in cont+['Degree']:
    g=[df.loc[df.seg==s,v] for s in (1,2,3)]; F,p=stats.f_oneway(*g)
    print(v,[round(x.mean(),2) for x in g],'overall',round(df[v].mean(),2),'F=%.1f p=%.2g'%(F,p))
# Tukey on price
t=stats.tukey_hsd(*[df.loc[df.seg==s,'Price'] for s in (1,2,3)]); print('Tukey Price p-values:\n',np.round(t.pvalue,4))
print('\nBINARY / CATEGORICAL')
cat=['iPhone','CompBuy','Female','Degree','AmznP','FB_Insta','Twit','Snap','YouTube','Pod_radio','TV','NewsP','Health','Finc','Sales','Advt','Edu','Cons','Eng','Tech','Retail','SMB','Other']
out=[]
for v in cat:
    ct=pd.crosstab(df.seg,df[v]); chi,p,_,_=stats.chi2_contingency(ct)
    V=np.sqrt(chi/(len(df)*(min(ct.shape)-1)))
    pct=df.groupby('seg')[v].apply(lambda s:(s==(2 if v=='Degree' else 1)).mean()*100)
    out.append((v,*pct.round(1).tolist(),round((df[v]==(2 if v=='Degree' else 1)).mean()*100,1),chi,p,V))
o=pd.DataFrame(out,columns=['var','S1%','S2%','S3%','All%','chi2','p','CramerV']); print(o.round(4).to_string(index=False))
# Income distribution
print(pd.crosstab(df.seg,df.Income,normalize='index').round(3))
ct=pd.crosstab(df.seg,df.Income); print('income chi2 p',stats.chi2_contingency(ct)[1])
# work: whole-variable test
df['Work']=df[['Health','Finc','Sales','Advt','Edu','Cons','Eng','Tech','Retail','SMB']].values.argmax(1)+1
df.loc[df.Other==1,'Work']=11
ct=pd.crosstab(df.seg,df.Work); chi,p,_,_=stats.chi2_contingency(ct); print('Work overall chi2=%.1f p=%.3g V=%.3f'%(chi,p,np.sqrt(chi/(1000*2))))
print((pd.crosstab(df.seg,df.Work,normalize='index')*100).round(1).to_string())
# price outlier robustness
print('Price Kruskal p',stats.kruskal(*[df.loc[df.seg==s,'Price'] for s in (1,2,3)])[1])
print('Age Kruskal p',stats.kruskal(*[df.loc[df.seg==s,'Age'] for s in (1,2,3)])[1])
print('Price by seg: median',df.groupby('seg').Price.median().to_dict(),' sd',df.groupby('seg').Price.std().round(1).to_dict(),' % >=$300',(df.assign(h=df.Price>=300).groupby('seg').h.mean()*100).round(1).to_dict())
df.to_pickle('out/df.pkl')
# profile chart
fig,ax=plt.subplots(1,2,figsize=(14,5))
ang=np.arange(12)
w=.27
for s in (1,2,3):
    ax[0].bar(ang+(s-2)*w,(means.loc[s]-df[rat].mean()).values,w,label=f'S{s}: {nm[order[s-1]]}')
ax[0].set_xticks(ang); ax[0].set_xticklabels(rat,rotation=45,ha='right'); ax[0].axhline(0,c='k',lw=.8); ax[0].legend(fontsize=7); ax[0].set_title('Segmentation variables: deviation from overall mean'); ax[0].set_ylabel('rating points')
dv=['iPhone','CompBuy','Female','AmznP','FB_Insta','Snap','TV','Twit','YouTube','Pod_radio','NewsP']
P=df.groupby('seg')[dv].mean()*100
for i,s in enumerate((1,2,3)): ax[1].bar(np.arange(len(dv))+(s-2)*w,P.loc[s].values,w,label=f'S{s}')
ax[1].set_xticks(range(len(dv))); ax[1].set_xticklabels(dv,rotation=45,ha='right'); ax[1].set_ylabel('% of segment'); ax[1].set_title('Descriptor profile (% yes)'); ax[1].legend()
plt.tight_layout(); plt.savefig('out/profile.png',dpi=130); plt.close()


# ======================================================================
# Q3/Q4 - ATTRACTIVENESS, PARTNER FIT, PRICE
# ======================================================================
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
df=pd.read_pickle('out/df.pkl')
rat=['Innov','ConstCom','CreatCom','TimelyInf','SaveM_T','SaveM_L','TaskMgm','DeviceSt','Photo','Wellness','Athlete','Style']
M=df.groupby('seg')[rat].mean()
sc=lambda x,lo,hi:np.clip(1+6*(x-lo)/(hi-lo),1,7)
S=pd.DataFrame(index=[1,2,3])
S['size_pct']=df.seg.value_counts(normalize=True).sort_index()*100
S['price']=df.groupby('seg').Price.mean()
S['inc_hi']=df.groupby('seg').Income.apply(lambda s:(s>=4).mean()*100)
S['need']=M.mean(axis=1)
S['fit']=M[['Wellness','Athlete','SaveM_L','Innov']].mean(axis=1)
S['access']=df.groupby('seg')[['FB_Insta','Twit','Snap','YouTube','Pod_radio','TV','NewsP','AmznP']].mean().mean(axis=1)*100
print(S.round(2).to_string())
A=pd.DataFrame(index=[1,2,3])
A['Size']=sc(S.size_pct,10,50); A['WTP']=sc(S.price,150,250); A['Income']=sc(S.inc_hi,10,60)
A['Need']=sc(S.need,3,5); A['Fit']=sc(S.fit,2,6)
# accessibility: metric is weakly differentiated; blend reach% with channel concentration judgement (documented)
A['Access']=[6,5,4]
W={'Need':.25,'WTP':.20,'Fit':.20,'Size':.15,'Income':.10,'Access':.10}
A['Overall']=sum(A[k]*w for k,w in W.items())
print(A.round(2).to_string()); 
# sensitivity
import itertools
eq=A[list(W)].mean(axis=1); print('equal-weight overall',eq.round(2).to_dict())
rng=np.random.default_rng(0); ranks=[]
for _ in range(5000):
    w=rng.dirichlet(np.ones(6)); ranks.append(A[list(W)].mul(w).sum(axis=1).rank(ascending=False).values)
r=np.array(ranks); print('P(rank1) by seg under random weights:',{s:round((r[:,i]==1).mean(),3) for i,s in enumerate([1,2,3])})
# partner fit
sets={'Google':['ConstCom','CreatCom','TimelyInf','Innov'],'Amazon':['TaskMgm','SaveM_T'],'Aetna':['SaveM_L','Wellness','Athlete']}
android=100-df.groupby('seg').iPhone.mean()*100; prime=df.groupby('seg').AmznP.mean()*100; cb=df.groupby('seg').CompBuy.mean()*100
eco={'Google':sc(android,20,70),'Amazon':sc(prime,30,80),'Aetna':sc(cb,0,60)}
F=pd.DataFrame({p:0.75*M[v].mean(axis=1)+0.25*eco[p] for p,v in sets.items()})
feat=pd.DataFrame({p:M[v].mean(axis=1) for p,v in sets.items()}); ecod=pd.DataFrame(eco)
dev=pd.DataFrame({p:(M[v]-df[rat].mean()[v]).mean(axis=1) for p,v in sets.items()})
print('feature fit');print(feat.round(2));print('ecosystem');print(ecod.round(2));print('distinctiveness (dev vs overall)');print(dev.round(2));print('FIT');print(F.round(2))
print('Android %',android.round(1).to_dict(),'Prime %',prime.round(1).to_dict(),'CompBuy %',cb.round(1).to_dict())
combo=(F.mul(A.Overall,axis=0)/7); print('combined (attractiveness x fit /7)');print(combo.round(2))
# price: S1 WTP / revenue curves
fig,ax=plt.subplots(1,2,figsize=(13,4.5))
for s in (1,2,3):
    p=df.loc[df.seg==s,'Price']; grid=np.arange(100,400,10)
    sh=np.array([(p>=g).mean() for g in grid]); ax[0].plot(grid,sh*100,label=f'S{s}')
    if s==1:
        rev=grid*sh; print('S1 revenue-max price (stated WTP, per-respondent):',grid[rev.argmax()],'share',round(sh[rev.argmax()]*100,1),
         ' | share>=199',round((p>=199).mean()*100,1),' >=229',round((p>=229).mean()*100,1),' >=249',round((p>=249).mean()*100,1),' pctls',p.quantile([.25,.5,.75]).tolist())
    if s==2: print('S2 pctls',p.quantile([.25,.5,.75,.9]).tolist(),' revmax', grid[(grid*np.array([(p>=g).mean() for g in grid])).argmax()])
ax[0].set_xlabel('Price point ($)'); ax[0].set_ylabel('% of segment with stated WTP ≥ price'); ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_title('Stated willingness-to-pay curves')
sns.heatmap(F,annot=True,fmt='.1f',cmap='YlGnBu',vmin=1,vmax=7,ax=ax[1]); ax[1].set_title('Segment × partner fit (1-7)'); ax[1].set_yticklabels(['S1 Social Wellness','S2 Always-On Pros','S3 Casual Snappers'],rotation=0)
plt.tight_layout(); plt.savefig('out/partner_price.png',dpi=130); plt.close()
# S1 detail
s1=df[df.seg==1]; print('S1 n',len(s1),'iPhone',round(s1.iPhone.mean()*100,1),'Prime',round(s1.AmznP.mean()*100,1),'CompBuy',round(s1.CompBuy.mean()*100,1),'Snap',round(s1.Snap.mean()*100,1),'FB',round(s1.FB_Insta.mean()*100,1),'inc>=101K',round((s1.Income>=4).mean()*100,1))
print('Overall Price>=300 share:',round((df.Price>=300).mean()*100,1))
print('Seg3 mean Style/Photo',M.loc[3,['Style','Photo']].round(2).to_dict())
print('S1 size n',len(s1),' S1 Prime & iPhone overlap',round(((s1.iPhone==1)&(s1.AmznP==1)).mean()*100,1))
A.to_pickle('out/A.pkl'); F.to_pickle('out/F.pkl')