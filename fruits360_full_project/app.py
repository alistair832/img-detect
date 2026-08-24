from __future__ import annotations
import base64, json, os, shutil, subprocess, sys, threading, time, urllib.parse, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent; APP=Path(__file__).resolve(); DATA=ROOT/'data'; MODELS=ROOT/'models'; OUT=ROOT/'outputs'
MODEL=MODELS/'best_fruit_model.keras'; CLASSES=MODELS/'class_names.json'; META=MODELS/'model_metadata.json'; CSV=OUT/'model_comparison.csv'; PNG=OUT/'model_comparison.png'
CFG={'dataset_handle':'moltean/fruits','image_size':128,'batch_size':32,'validation_split':.2,'seed':123,'cnn_epochs':8,'transfer_epochs':8,'finetune_epochs':4,'deployment_max_mb':90.0}
RULES=[('Pomegranate',('pomegranate',)),('Pineapple',('pineapple',)),('Dragon Fruit',('dragon fruit','pitahaya')),('Passion Fruit',('passion fruit','maracuja')),('Grapefruit',('grapefruit',)),('Watermelon',('watermelon',)),('Cantaloupe',('cantaloupe',)),('Mandarin',('mandarin','mandarine','clementine')),('Coconut',('coconut','cocos')),('Blackberry',('blackberry',)),('Blueberry',('blueberry',)),('Raspberry',('raspberry',)),('Strawberry',('strawberry',)),('Apple',('apple',)),('Banana',('banana',)),('Orange',('orange',)),('Lemon',('lemon',)),('Lime',('lime','limetta')),('Mango',('mango',)),('Papaya',('papaya',)),('Avocado',('avocado',)),('Apricot',('apricot',)),('Peach',('peach',)),('Nectarine',('nectarine',)),('Pear',('pear',)),('Plum',('plum',)),('Cherry',('cherry',)),('Grape',('grape',)),('Kiwi',('kiwi',)),('Guava',('guava',)),('Lychee',('lychee','litchi')),('Fig',('fig',)),('Persimmon',('persimmon','kaki')),('Carambola',('carambola','star fruit')),('Melon',('melon',)),('Tomato',('tomato',)),('Pepper',('pepper','capsicum')),('Cucumber',('cucumber',)),('Eggplant',('eggplant','aubergine')),('Potato',('potato',)),('Onion',('onion',)),('Garlic',('garlic',)),('Carrot',('carrot',)),('Beetroot',('beetroot','beet')),('Cabbage',('cabbage',)),('Cauliflower',('cauliflower',)),('Broccoli',('broccoli',)),('Corn',('corn',)),('Ginger',('ginger',)),('Hazelnut',('hazelnut',)),('Walnut',('walnut',)),('Almond',('almond',)),('Chestnut',('chestnut',))]

def generic(name):
    n=' '.join(name.lower().replace('_',' ').replace('-',' ').split())
    for label,keys in RULES:
        if any(k in n for k in keys): return label
    return name

def groups(names):
    order=[]; d={}
    for i,n in enumerate(names):
        g=generic(n)
        if g not in d: d[g]=[]; order.append(g)
        d[g].append(i)
    return order,[d[x] for x in order]

# ---------- isolated training worker ----------
def worker_imports():
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2'); os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS','0'); os.environ.setdefault('OMP_NUM_THREADS','2')
    import kagglehub, matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt, numpy as np, pandas as pd, tensorflow as tf
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    return kagglehub,plt,np,pd,tf,accuracy_score,precision_recall_fscore_support

def split(root,name):
    c=[p for p in Path(root).rglob(name) if p.is_dir()]
    if not c: raise FileNotFoundError(f'No {name} below {root}')
    c.sort(key=lambda p:('100x100' in str(p).lower(),sum(x.is_dir() for x in p.iterdir())),reverse=True); return c[0]

def datasets(tf,tr,te):
    s=(CFG['image_size'],)*2; kw=dict(image_size=s,batch_size=CFG['batch_size'],label_mode='int')
    a=tf.keras.utils.image_dataset_from_directory(tr,validation_split=CFG['validation_split'],subset='training',seed=CFG['seed'],shuffle=True,**kw)
    b=tf.keras.utils.image_dataset_from_directory(tr,validation_split=CFG['validation_split'],subset='validation',seed=CFG['seed'],shuffle=False,**kw)
    c=tf.keras.utils.image_dataset_from_directory(te,shuffle=False,**kw); names=list(a.class_names); au=tf.data.AUTOTUNE
    return a.prefetch(au),b.prefetch(au),c.prefetch(au),names,s

def aug(tf): return tf.keras.Sequential([tf.keras.layers.RandomFlip('horizontal'),tf.keras.layers.RandomRotation(.15),tf.keras.layers.RandomZoom(.12),tf.keras.layers.RandomContrast(.15),tf.keras.layers.RandomTranslation(.06,.06)])
def compile_model(tf,m,lr): m.compile(optimizer=tf.keras.optimizers.Adam(lr),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
def cnn(tf,n,s):
    i=tf.keras.Input(shape=s+(3,)); x=aug(tf)(i); x=tf.keras.layers.Rescaling(1/255)(x)
    for f in (32,64,128): x=tf.keras.layers.Conv2D(f,3,padding='same',activation='relu')(x); x=tf.keras.layers.BatchNormalization()(x); x=tf.keras.layers.MaxPooling2D()(x)
    x=tf.keras.layers.GlobalAveragePooling2D()(x); x=tf.keras.layers.Dropout(.3)(x); m=tf.keras.Model(i,tf.keras.layers.Dense(n)(x),name='Custom_CNN'); compile_model(tf,m,1e-3); return m

def transfer(tf,name,n,s):
    if name=='MobileNetV2': base=tf.keras.applications.MobileNetV2(include_top=False,weights='imagenet',input_shape=s+(3,)); pre=tf.keras.applications.mobilenet_v2.preprocess_input
    else: base=tf.keras.applications.ResNet50(include_top=False,weights='imagenet',input_shape=s+(3,)); pre=tf.keras.applications.resnet50.preprocess_input
    base.trainable=False; i=tf.keras.Input(shape=s+(3,)); x=base(pre(aug(tf)(i)),training=False); x=tf.keras.layers.GlobalAveragePooling2D()(x); x=tf.keras.layers.Dropout(.25)(x); m=tf.keras.Model(i,tf.keras.layers.Dense(n)(x),name=name); compile_model(tf,m,1e-3); return m,base

def fit(tf,m,tr,va,e,quick):
    kw=dict(validation_data=va,epochs=e,verbose=2,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=2,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.3,patience=1,min_lr=1e-7)])
    if quick: kw.update(steps_per_epoch=30,validation_steps=10)
    m.fit(tr,**kw)

def evaluate(tf,np,acc_fn,prf,m,ds,quick):
    yt=[]; yp=[]; start=time.perf_counter()
    for j,(x,y) in enumerate(ds): yt.extend(y.numpy().astype(int)); yp.extend(tf.argmax(m(x,training=False),1).numpy().astype(int));
    acc=acc_fn(yt,yp); p,r,f,_=prf(yt,yp,average='macro',zero_division=0); sec=time.perf_counter()-start
    return float(acc),float(p),float(r),float(f),float(sec),len(yt)/sec if sec else 0

def training_worker(quick=False):
    if sys.version_info[:2]!=(3,12): raise RuntimeError('Use Python 3.12')
    kagglehub,plt,np,pd,tf,acc_fn,prf=worker_imports(); tf.keras.utils.set_random_seed(CFG['seed']); MODELS.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)
    root=Path(kagglehub.dataset_download(CFG['dataset_handle'],output_dir=str(DATA))); tr=split(root,'Training'); te=split(root,'Test'); train,val,test,names,size=datasets(tf,tr,te); n=len(names)
    models={}; epochs=(1,1,1) if quick else (CFG['cnn_epochs'],CFG['transfer_epochs'],CFG['finetune_epochs'])
    m=cnn(tf,n,size); fit(tf,m,train,val,epochs[0],quick); m.save(MODELS/'custom_cnn.keras'); models['Custom_CNN']=m
    for name,last,path in [('MobileNetV2',30,'mobilenetv2.keras'),('ResNet50',25,'resnet50.keras')]:
        m,b=transfer(tf,name,n,size); fit(tf,m,train,val,epochs[1],quick); b.trainable=True
        for layer in b.layers[:-last]: layer.trainable=False
        compile_model(tf,m,1e-5); fit(tf,m,train,val,epochs[2],quick); m.save(MODELS/path); models[name]=m
    rows=[]; paths={'Custom_CNN':MODELS/'custom_cnn.keras','MobileNetV2':MODELS/'mobilenetv2.keras','ResNet50':MODELS/'resnet50.keras'}
    for name,m in models.items():
        a,p,r,f,s,ips=evaluate(tf,np,acc_fn,prf,m,test,quick); rows.append({'Model':name,'Accuracy':a,'Macro Precision':p,'Macro Recall':r,'Macro F1':f,'Inference seconds':s,'Images / second':ips,'Model MB':paths[name].stat().st_size/1048576})
    df=pd.DataFrame(rows).sort_values('Macro F1',ascending=False).reset_index(drop=True); df.to_csv(CSV,index=False); ax=df.set_index('Model')[['Accuracy','Macro Precision','Macro Recall','Macro F1']].plot(kind='bar',figsize=(10,5)); ax.set_ylim(0,1.05); plt.tight_layout(); plt.savefig(PNG,dpi=160); plt.close()
    eligible=df[df['Model MB']<=CFG['deployment_max_mb']]; row=(eligible.iloc[0] if len(eligible) else df.iloc[0]); name=str(row['Model']); shutil.copy2(paths[name],MODEL); CLASSES.write_text(json.dumps(names,ensure_ascii=False,indent=2),encoding='utf-8'); gn,_=groups(names)
    META.write_text(json.dumps({'dataset_handle':CFG['dataset_handle'],'best_model':name,'num_classes':len(names),'num_generic_classes':len(gn),'test_accuracy':float(row['Accuracy']),'macro_f1':float(row['Macro F1']),'quick_run':bool(quick),'tensorflow_version':tf.__version__},indent=2),encoding='utf-8')

if '--train-worker' in sys.argv: training_worker('--quick' in sys.argv); raise SystemExit

# ---------- Streamlit ----------
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2'); os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS','0'); os.environ.setdefault('OMP_NUM_THREADS','1')
import av, numpy as np, streamlit as st
from PIL import Image,ImageDraw
from streamlit_webrtc import webrtc_streamer
st.set_page_config(page_title='Fruits-360 AI System',page_icon='🍎',layout='wide')

def secret(k):
    if os.environ.get(k): return os.environ[k]
    try: return str(st.secrets.get(k,'') or '')
    except Exception: return ''

def metadata():
    try: return json.loads(META.read_text(encoding='utf-8')) if META.exists() else {}
    except Exception: return {}

def ready(): return MODEL.exists() and CLASSES.exists()

@st.cache_data(ttl=2400,show_spinner=False)
def rtc_config():
    fallback={'iceServers':[{'urls':['stun:stun.l.google.com:19302','stun:stun1.l.google.com:19302','stun:stun2.l.google.com:19302']},{'urls':['stun:stun.cloudflare.com:3478']} ]}
    sid,token=secret('TWILIO_ACCOUNT_SID'),secret('TWILIO_AUTH_TOKEN')
    if not sid or not token: return fallback,False,'TURN credentials not configured.'
    try:
        req=urllib.request.Request(f"https://api.twilio.com/2010-04-01/Accounts/{urllib.parse.quote(sid,safe='')}/Tokens.json",data=b'',method='POST'); auth=base64.b64encode(f'{sid}:{token}'.encode()).decode(); req.add_header('Authorization',f'Basic {auth}'); req.add_header('Content-Type','application/x-www-form-urlencoded')
        with urllib.request.urlopen(req,timeout=10) as r: payload=json.loads(r.read().decode())
        ice=payload.get('ice_servers') or payload.get('iceServers')
        if not ice: raise RuntimeError('No ICE servers returned')
        return {'iceServers':ice},True,'TURN relay enabled.'
    except Exception as e: return fallback,False,f'TURN failed; STUN fallback: {e}'

st.title('🍎 Fruits-360 All-in-One AI System'); meta=metadata(); model_ready=ready(); a,b,c,d=st.columns(4); a.metric('Python',sys.version.split()[0]); b.metric('Model','Ready ✅' if model_ready else 'Not trained'); c.metric('Classes',meta.get('num_classes','—')); d.metric('Accuracy',f"{meta.get('test_accuracy',0)*100:.2f}%" if 'test_accuracy' in meta else '—')
train_tab,result_tab,detect_tab=st.tabs(['1️⃣ Train Model','2️⃣ Results','3️⃣ Detect Fruit'])
with train_tab:
    tok=secret('KAGGLE_API_TOKEN'); entered='' if tok else st.text_input('Kaggle API token',type='password'); mode=st.radio('Training mode',['Quick pipeline test','Full assignment training'],horizontal=True); st.warning('Quick mode is not for final recognition.') if mode.startswith('Quick') else st.info('Full mode uses the selected Fruits-360 training split.')
    if st.button('🚀 Start Training',disabled=not bool(tok or entered),type='primary',use_container_width=True):
        env=os.environ.copy(); env['KAGGLE_API_TOKEN']=entered or tok; env['PYTHONUNBUFFERED']='1'; cmd=[sys.executable,str(APP),'--train-worker']+(['--quick'] if mode.startswith('Quick') else []); box=st.empty(); status=st.status('Training started...',expanded=True); lines=[]
        try:
            p=subprocess.Popen(cmd,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env)
            for line in p.stdout: lines=(lines+[line.rstrip()])[-80:]; box.code('\n'.join(lines),language='text')
            rc=p.wait(); status.update(label='Training completed' if rc==0 else f'Training stopped ({rc})',state='complete' if rc==0 else 'error');
            if rc==0: st.cache_resource.clear(); time.sleep(1); st.rerun()
        except Exception as e: status.update(label='Could not start training',state='error'); st.exception(e)
with result_tab:
    if CSV.exists():
        import pandas as pd
        df=pd.read_csv(CSV); show=df.copy();
        for x in ['Accuracy','Macro Precision','Macro Recall','Macro F1']: show[x]=show[x].map(lambda v:f'{v*100:.2f}%') if x in show else show.get(x)
        st.dataframe(show,use_container_width=True,hide_index=True); st.image(str(PNG),use_container_width=True) if PNG.exists() else None
    else: st.info('No results yet.')
with detect_tab:
    if not ready(): st.info('Train a full model first.')
    elif meta.get('quick_run'): st.error('Recognition is disabled because this is a Quick Test model.')
    else:
        import tensorflow as tf
        @st.cache_resource(show_spinner='Loading model...')
        def runtime():
            m=tf.keras.models.load_model(MODEL,compile=False); names=json.loads(CLASSES.read_text(encoding='utf-8')); gn,gg=groups(names); return m,names,gn,gg,(int(m.input_shape[2]),int(m.input_shape[1]))
        model,names,gn,gg,img_size=runtime()
        def probs(img):
            arr=np.asarray(img.convert('RGB').resize(img_size,Image.Resampling.BILINEAR),dtype=np.float32)[None,...]; detail=tf.nn.softmax(model(arr,training=False)[0]).numpy().astype(np.float32); g=np.array([detail[ix].sum() for ix in gg],dtype=np.float32); g/=g.sum() if g.sum() else 1; return detail,g
        def predict(img,k=5):
            detail,g=probs(img); oi=np.argsort(g)[::-1][:k]; di=np.argsort(detail)[::-1][:k]; return [(gn[int(i)],float(g[int(i)])) for i in oi],[(names[int(i)],float(detail[int(i)])) for i in di]
        INTERVAL=.35; ALPHA=.5; MIN=.38; MARGIN=.03
        if '_cam_v5' not in globals(): _cam_v5={'t':0.0,'p':None,'label':'Analyzing fruit...'}
        lock=threading.Lock()
        def update(img):
            _,g=probs(img); s=_cam_v5['p']; s=g if s is None or len(s)!=len(g) else ALPHA*g+(1-ALPHA)*s; _cam_v5['p']=s; order=np.argsort(s)[::-1]; i,j=int(order[0]),int(order[1]); conf=float(s[i]); _cam_v5['label']=f'{gn[i]} — {conf*100:.1f}%' if conf>=MIN and conf-float(s[j])>=MARGIN else 'Unknown / hold fruit steady'; _cam_v5['t']=time.monotonic()
        def callback(frame):
            rgb=frame.to_ndarray(format='rgb24')[:,::-1].copy(); img=Image.fromarray(rgb); w,h=img.size; z=int(min(w,h)*.6); l=(w-z)//2; t=(h-z)//2; r=l+z; bot=t+z
            if time.monotonic()-_cam_v5['t']>=INTERVAL and lock.acquire(False):
                try: update(img.crop((l,t,r,bot)))
                except Exception: pass
                finally: lock.release()
            draw=ImageDraw.Draw(img); draw.rectangle((l,t,r,bot),outline=(40,220,90),width=4); draw.rectangle((12,12,max(330,w-12),62),fill=(0,0,0)); draw.text((22,27),_cam_v5['label'],fill=(255,255,255)); return av.VideoFrame.from_ndarray(np.asarray(img),format='rgb24')
        live,snapshot,upload=st.tabs(['🎥 Live Camera','📸 Camera Photo','🖼️ Upload'])
        with live:
            rtc,turn,msg=rtc_config(); st.success('TURN relay enabled ✅') if turn else st.warning('STUN fallback only. If START turns off, add Twilio TURN credentials in Streamlit Secrets.')
            if not turn:
                with st.expander('TURN setup'): st.code('TWILIO_ACCOUNT_SID = "AC..."\nTWILIO_AUTH_TOKEN = "..."',language='toml'); st.caption(msg)
            st.caption('Allow browser camera permission, then press START.')
            webrtc_streamer(key='fruits360-live-v5',video_frame_callback=callback,media_stream_constraints={'video':{'facingMode':'user','width':{'ideal':480},'height':{'ideal':360},'frameRate':{'ideal':20,'max':24}},'audio':False},rtc_configuration=rtc,async_processing=True)
        with snapshot:
            st.info('Reliable fallback: this does not use WebRTC.')
            cap=st.camera_input('Take a photo of one fruit')
            if cap:
                image=Image.open(cap).convert('RGB'); res,_=predict(image); st.success(f'Detected: **{res[0][0]}**'); st.metric('Confidence',f'{res[0][1]*100:.2f}%'); [st.write(f'**{n}** — {p*100:.2f}%') for n,p in res]
        with upload:
            up=st.file_uploader('Upload JPG, PNG or WEBP',type=['jpg','jpeg','png','webp'])
            if up:
                image=Image.open(up).convert('RGB'); res,detail=predict(image); st.image(image,use_container_width=True); st.success(f'Detected: **{res[0][0]}**'); st.metric('Confidence',f'{res[0][1]*100:.2f}%'); [st.write(f'**{n}** — {p*100:.2f}%') for n,p in res]
                with st.expander('Detailed Fruits-360 classes'): [st.write(f'{n} — {p*100:.2f}%') for n,p in detail]
st.divider(); st.caption('Fruits-360 recognition with TURN/STUN live camera and a reliable non-WebRTC camera fallback.')
