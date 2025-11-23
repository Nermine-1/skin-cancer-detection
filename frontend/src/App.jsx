import React, {useState, useEffect} from 'react'
import { motion, AnimatePresence } from 'framer-motion'

function TopBar({label, confidence}){
  return (
    <motion.div className="topbar" initial={{opacity:0}} animate={{opacity:1}}>
      <div className="top-title">Prediction</div>
      <div className="top-meta">{label ?? '—'} · {(confidence*100 || 0).toFixed(1)}%</div>
    </motion.div>
  )
}

function ProbBar({label, value, i}){
  return (
    <motion.div className="prob-row" initial={{opacity:0, x:-20}} animate={{opacity:1, x:0}} transition={{delay: i*0.06}}>
      <div className="prob-label">{label}</div>
      <div className="prob-track">
        <motion.div className="prob-fill" initial={{width:0}} animate={{width: `${Math.max(2, value*100)}%`}} transition={{duration:0.6}} />
      </div>
      <div className="prob-val">{(value*100).toFixed(1)}%</div>
    </motion.div>
  )
}

export default function App(){
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [showOnboard, setShowOnboard] = useState(true)

  useEffect(()=>{
    return ()=>{ if(preview) URL.revokeObjectURL(preview) }
  }, [preview])

  function handleFile(e){
    const f = e.target.files[0]
    if(!f) return
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
  }

  async function submit(){
    if(!file) return
    setLoading(true)
    const form = new FormData()
    form.append('file', file)
    try{
      const res = await fetch((process.env.REACT_APP_API_URL ?? 'http://localhost:8000') + '/predict', {method: 'POST', body: form})
      const data = await res.json()
      setResult(data)
    }catch(err){
      setResult({error: err.message})
    }finally{
      setLoading(false)
    }
  }

  return (
    <div className="container creative">
      <AnimatePresence>
        {showOnboard && (
          <motion.div className="onboard" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}}>
            <motion.h2 initial={{y:-20}} animate={{y:0}} transition={{delay:0.05}}>Welcome to SkinScope</motion.h2>
            <motion.p initial={{y:-10, opacity:0}} animate={{y:0, opacity:1}} transition={{delay:0.12}}>Drag an image or click to upload — watch the model explain its prediction with animated insights.</motion.p>
            <motion.button onClick={()=>setShowOnboard(false)} whileHover={{scale:1.03}} whileTap={{scale:0.98}}>Get started</motion.button>
          </motion.div>
        )}
      </AnimatePresence>

      <h1>SkinScope</h1>

      <div className="uploader">
        <label className="file-input">
          <input type="file" accept="image/*" onChange={handleFile} />
          <div className="drop-area">{preview ? 'Change image' : 'Click or drop an image'}</div>
        </label>

        <div className="preview-area">
          {preview ? (
            <motion.img src={preview} alt="preview" className="preview" initial={{opacity:0, scale:0.9}} animate={{opacity:1, scale:1}} transition={{duration:0.4}}/>
          ) : (
            <div className="placeholder">No image</div>
          )}
        </div>

        <div className="controls">
          <button className="analyze" onClick={submit} disabled={loading || !file}>{loading ? 'Analyzing…' : 'Analyze'}</button>
          <button className="clear" onClick={()=>{setFile(null); setPreview(null); setResult(null)}}>Clear</button>
        </div>
      </div>

      <AnimatePresence>
        {result && (
          <motion.div className="result" initial={{y:18, opacity:0}} animate={{y:0, opacity:1}} exit={{y:18, opacity:0}}>
            {result.error ? (
              <div className="error">Error: {result.error}</div>
            ) : (
              <>
                <TopBar label={result.pred_label ?? `Class ${result.pred_index}`} confidence={result.confidence} />

                <motion.div className="result-body" initial={{opacity:0}} animate={{opacity:1}}>
                  <div className="result-left">
                    <div className="big-label">{result.pred_label ?? `Class ${result.pred_index}`}</div>
                    <div className="big-conf">{(result.confidence*100).toFixed(1)}%</div>
                  </div>
                  <div className="result-right">
                    <h4>Probabilities</h4>
                    <div className="probs">
                      {Array.isArray(result.probabilities) && (result.classes || Array.from({length: result.probabilities.length}, (_,i)=>`Class ${i}`)).map((lab, i)=> (
                        <ProbBar key={i} label={(result.classes||[])[i] ?? lab} value={result.probabilities[i]} i={i} />
                      ))}
                    </div>
                  </div>
                </motion.div>
              </>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
