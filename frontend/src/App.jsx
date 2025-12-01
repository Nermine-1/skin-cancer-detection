import React, {useState, useEffect} from 'react'
import { motion, AnimatePresence } from 'framer-motion'

// Background particles component
function BackgroundEffects() {
  const particles = Array.from({ length: 20 }, (_, i) => ({
    id: i,
    size: Math.random() * 6 + 2,
    left: Math.random() * 100,
    delay: Math.random() * 20,
  }))

  const shapes = Array.from({ length: 8 }, (_, i) => ({
    id: i,
    type: ['triangle', 'square', 'circle'][Math.floor(Math.random() * 3)],
    left: Math.random() * 100,
    delay: Math.random() * 25,
  }))

  return (
    <div className="particles">
      {particles.map((particle) => (
        <div
          key={particle.id}
          className="particle"
          style={{
            width: `${particle.size}px`,
            height: `${particle.size}px`,
            left: `${particle.left}%`,
            animationDelay: `${particle.delay}s`,
          }}
        />
      ))}
      {shapes.map((shape) => (
        <div
          key={`shape-${shape.id}`}
          className={`geometric-shape ${shape.type}`}
          style={{
            left: `${shape.left}%`,
            animationDelay: `${shape.delay}s`,
          }}
        />
      ))}
    </div>
  )
}

function TopBar({label, confidence}){
  return (
    <motion.div className="topbar" initial={{opacity:0}} animate={{opacity:1}}>
      <div className="top-title">Prediction</div>
      <div className="top-meta">{label ?? '—'} · {(confidence*100 || 0).toFixed(1)}%</div>
    </motion.div>
  )
}

function ProbBar({label, value, i, isTopPrediction}){
  const percentage = value * 100;
  const getBarColor = () => {
    if (percentage >= 70) return 'var(--success)';
    if (percentage >= 30) return 'var(--warning)';
    return 'var(--error)';
  };

  return (
    <motion.div
      className={`prob-row ${isTopPrediction ? 'top-prediction' : ''}`}
      initial={{opacity:0, x:-20, scale: 0.95}}
      animate={{opacity:1, x:0, scale: 1}}
      transition={{
        delay: i*0.08,
        duration: 0.5,
        type: "spring",
        stiffness: 100
      }}
      whileHover={{ scale: 1.02 }}
    >
      <div className="prob-label">{label}</div>
      <div className="prob-track">
        <motion.div
          className="prob-fill"
          initial={{width:0}}
          animate={{
            width: `${Math.max(2, percentage)}%`,
            background: `linear-gradient(90deg, ${getBarColor()}, ${getBarColor()}dd)`
          }}
          transition={{
            duration: 1.2,
            delay: i*0.08 + 0.2,
            ease: "easeOut"
          }}
        />
        <motion.div
          className="prob-glow"
          initial={{opacity: 0}}
          animate={{opacity: percentage > 50 ? 0.3 : 0}}
          transition={{delay: i*0.08 + 0.5, duration: 0.3}}
        />
      </div>
      <motion.div
        className="prob-val"
        initial={{opacity: 0, scale: 0.8}}
        animate={{opacity: 1, scale: 1}}
        transition={{delay: i*0.08 + 0.8, duration: 0.3}}
      >
        {percentage.toFixed(1)}%
      </motion.div>
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
      <BackgroundEffects />

      {/* Loading Overlay */}
      <AnimatePresence>
        {loading && (
          <motion.div
            className="loading-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className="loading-content">
              <div className="loading-spinner-large"></div>
              <div className="loading-text">Analyzing Image</div>
              <div className="loading-subtext">AI is processing your skin image...</div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {showOnboard && (
          <motion.div className="onboard" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}}>
            <motion.div className="onboard-content" initial={{scale:0.9, opacity:0}} animate={{scale:1, opacity:1}} transition={{delay:0.1, duration:0.5}}>
              <motion.h2
                initial={{y:-30, opacity:0}}
                animate={{y:0, opacity:1}}
                transition={{delay:0.2, duration:0.6}}
              >
                Welcome to SkinScope
              </motion.h2>
              <motion.p
                initial={{y:-20, opacity:0}}
                animate={{y:0, opacity:1}}
                transition={{delay:0.4, duration:0.6}}
              >
                Advanced AI-powered skin cancer detection with real-time analysis
              </motion.p>

              <motion.div
                className="features"
                initial={{opacity:0}}
                animate={{opacity:1}}
                transition={{delay:0.6, duration:0.6}}
              >
                <motion.div
                  className="feature-item"
                  initial={{x:-20, opacity:0}}
                  animate={{x:0, opacity:1}}
                  transition={{delay:0.7, duration:0.4}}
                >
                  <div className="feature-icon">🔬</div>
                  <span>AI-Powered Analysis</span>
                </motion.div>
                <motion.div
                  className="feature-item"
                  initial={{x:-20, opacity:0}}
                  animate={{x:0, opacity:1}}
                  transition={{delay:0.8, duration:0.4}}
                >
                  <div className="feature-icon">⚡</div>
                  <span>Instant Results</span>
                </motion.div>
                <motion.div
                  className="feature-item"
                  initial={{x:-20, opacity:0}}
                  animate={{x:0, opacity:1}}
                  transition={{delay:0.9, duration:0.4}}
                >
                  <div className="feature-icon">📊</div>
                  <span>Detailed Probabilities</span>
                </motion.div>
              </motion.div>

              <motion.button
                onClick={()=>setShowOnboard(false)}
                whileHover={{scale:1.05, boxShadow: "0 8px 25px rgba(139, 92, 246, 0.4)"}}
                whileTap={{scale:0.95}}
                initial={{y:20, opacity:0}}
                animate={{y:0, opacity:1}}
                transition={{delay:1.0, duration:0.4}}
              >
                Start Analyzing
              </motion.button>
            </motion.div>
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
          <button className="analyze" onClick={submit} disabled={loading || !file}>
            {loading ? (
              <>
                <div className="loading-spinner"></div>
                Analyzing…
              </>
            ) : (
              'Analyze'
            )}
          </button>
          <button className="clear" onClick={()=>{setFile(null); setPreview(null); setResult(null)}} disabled={loading}>Clear</button>
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
                        <ProbBar
                          key={i}
                          label={(result.classes||[])[i] ?? lab}
                          value={result.probabilities[i]}
                          i={i}
                          isTopPrediction={i === 0 && result.probabilities[i] === Math.max(...result.probabilities)}
                        />
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
