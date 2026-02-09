import { useState, useEffect, useRef, useCallback } from 'react'
import { Upload, ArrowRight, ArrowLeft, MessageSquarePlus, Expand, Loader2, X, ZoomIn, Send } from 'lucide-react'
import * as THREE from 'three'
import gsap from 'gsap'
import './App.css'

const API_BASE_URL = 'http://localhost:8000'

// ============ UTILITY FUNCTIONS ============

function formatFieldName(key) {
  return key
    .replace(/_/g, ' ')
    .replace(/([A-Z])/g, ' $1')
    .trim()
}

// Parse text with *asterisks* or **double asterisks** and render as bold
function formatTextWithBold(text) {
  if (!text || typeof text !== 'string') return text

  // First handle **double asterisks**, then *single asterisks*
  // Pattern matches **text** or *text*
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g)

  return parts.map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      // Remove double asterisks and wrap in strong tag
      return <strong key={index}>{part.slice(2, -2)}</strong>
    }
    if (part.startsWith('*') && part.endsWith('*') && part.length > 2) {
      // Remove single asterisks and wrap in strong tag
      return <strong key={index}>{part.slice(1, -1)}</strong>
    }
    return part
  })
}

// ============ GLASSMORPHISM CARD ============

function GlassCard({ label, children, span = '', featured = false, delay = 0 }) {
  const cardRef = useRef(null)

  useEffect(() => {
    if (cardRef.current) {
      gsap.fromTo(cardRef.current,
        { opacity: 0, y: 20 },
        { opacity: 1, y: 0, duration: 0.6, delay: delay * 0.08, ease: 'power2.out' }
      )
    }
  }, [delay])

  return (
    <div
      ref={cardRef}
      className={`glass-panel ${featured ? 'glass-panel-featured' : ''} ${span}`}
    >
      <div className="label-text">{label}</div>
      <div className="card-content">
        {children}
      </div>
    </div>
  )
}

// ============ PATIENT HERO SECTION ============

function PatientHero({ patientInfo, reportDate }) {
  const name = patientInfo?.name || 'Patient Report'
  const age = patientInfo?.age || '—'
  const sex = patientInfo?.sex || '—'

  return (
    <section className="hero-section">
      <div className="hero-status">
        <span className="ping-dot" />
        <span className="status-text">Extraction Complete</span>
      </div>

      <h2 className="hero-title">{name}</h2>

      <div className="hero-meta">
        <div className="meta-item">
          <div className="meta-label">Patient Info</div>
          <div className="meta-value">{age} / {sex}</div>
        </div>
        <div className="meta-item">
          <div className="meta-label">Date</div>
          <div className="meta-value">{reportDate || new Date().toLocaleDateString()}</div>
        </div>
        <div className="meta-item">
          <div className="meta-label">Source</div>
          <div className="meta-value">Uploaded PDF</div>
        </div>
      </div>
    </section>
  )
}

// ============ DIAGNOSIS CARD (FEATURED) ============

function DiagnosisCard({ diagnosis, delay }) {
  if (!diagnosis) return null

  return (
    <GlassCard label="Primary Diagnosis" span="col-span-2" featured delay={delay}>
      <span className="diagnosis-text">{diagnosis}</span>
    </GlassCard>
  )
}

// ============ TEXT CONTENT CARD ============

function TextCard({ label, content, delay, span = '' }) {
  if (!content) return null

  // Determine span based on content length
  const autoSpan = content.length > 200 ? 'col-span-2' : span

  return (
    <GlassCard label={formatFieldName(label)} span={autoSpan} delay={delay}>
      <p className="text-content">{formatTextWithBold(content)}</p>
    </GlassCard>
  )
}

// ============ MEDICATIONS CARD ============

function MedicationsCard({ medications, delay }) {
  if (!medications || medications.length === 0) return null

  return (
    <GlassCard label="Medications" delay={delay}>
      <div className="medications-list">
        {medications.map((med, i) => {
          const name = med.name || 'Medication'
          const details = [med.dosage, med.frequency, med.duration].filter(Boolean).join(' • ')
          return (
            <div key={i} className="medication-item">
              <span className="med-name">{name}</span>
              <span className="med-details">{details}</span>
            </div>
          )
        })}
      </div>
    </GlassCard>
  )
}

// ============ KEY-VALUE CARD ============

function KeyValueCard({ label, data, delay }) {
  if (!data || typeof data !== 'object') return null

  const entries = Object.entries(data).filter(([_, v]) => v !== null && v !== undefined)
  if (entries.length === 0) return null

  return (
    <GlassCard label={formatFieldName(label)} delay={delay}>
      <div className="kv-list">
        {entries.map(([key, value]) => (
          <div key={key} className="kv-item">
            <span className="kv-key">{formatFieldName(key)}</span>
            <span className="kv-value">{String(value)}</span>
          </div>
        ))}
      </div>
    </GlassCard>
  )
}

// ============ DATA TABLE ============

function DataTableCard({ tableData, reportMetadata, delay }) {
  const { table_name, columns = [], rows = [], value_explanations = {} } = tableData
  const [activeTooltip, setActiveTooltip] = useState(null)
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 })
  const [tooltipType, setTooltipType] = useState(null) // 'explanation' or 'metadata'

  // Filter out duplicate rows - keep only unique rows based on content
  const uniqueRows = rows.filter((row, index, self) =>
    index === self.findIndex(r => JSON.stringify(r) === JSON.stringify(row))
  )

  if (uniqueRows.length === 0) return null

  const colCount = columns.length
  const span = colCount > 3 ? 'col-span-2' : colCount > 5 ? 'col-span-3' : ''
  const gridStyle = { gridTemplateColumns: `2fr repeat(${Math.max(1, colCount - 1)}, minmax(100px, 1fr))` }

  // Check if a row has an explanation
  const getRowExplanation = (rowIndex) => {
    return value_explanations[String(rowIndex)] || null
  }

  // Check if metadata has any non-null values
  const hasMetadata = reportMetadata && Object.values(reportMetadata).some(v => v !== null && v !== undefined)

  const TOOLTIP_WIDTH = 340 // Width of tooltip + some margin

  const handleMouseEnter = (e, data, type) => {
    const rect = e.target.getBoundingClientRect()
    const windowWidth = window.innerWidth

    // Check if there's enough space on the right
    const spaceOnRight = windowWidth - rect.right
    const positionLeft = spaceOnRight < TOOLTIP_WIDTH

    setTooltipPosition({
      x: positionLeft ? rect.left - 10 : rect.right + 10,
      y: rect.top + rect.height / 2,
      side: positionLeft ? 'left' : 'right'
    })
    setActiveTooltip(data)
    setTooltipType(type)
  }

  const handleMouseLeave = () => {
    setActiveTooltip(null)
    setTooltipType(null)
  }

  return (
    <>
      <GlassCard label={table_name || 'Data Table'} span={span} delay={delay}>
        {/* Metadata info icon */}
        {hasMetadata && (
          <div
            className="metadata-icon"
            onMouseEnter={(e) => handleMouseEnter(e, reportMetadata, 'metadata')}
            onMouseLeave={handleMouseLeave}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="12" y1="16" x2="12" y2="12"></line>
              <line x1="12" y1="8" x2="12.01" y2="8"></line>
            </svg>
          </div>
        )}
        <div className="data-table-wrapper">
          <div className="data-table">
            {/* Header */}
            <div className="table-header" style={gridStyle}>
              {columns.map((col, i) => <div key={i} className="table-header-cell">{col}</div>)}
            </div>
            {/* Rows */}
            {uniqueRows.map((row, ri) => {
              const explanation = getRowExplanation(ri)
              const hasExplanation = explanation && explanation.explanation

              return (
                <div
                  key={ri}
                  className={`table-row ${hasExplanation ? 'table-row-abnormal' : ''}`}
                  style={gridStyle}
                >
                  {row.map((cell, ci) => (
                    <div key={ci} className="table-cell">
                      {ci === 0 ? (
                        <div className="cell-with-tooltip">
                          <span className={`data-tag ${hasExplanation ? 'data-tag-abnormal' : ''}`}>
                            {cell}
                          </span>
                          {hasExplanation && (
                            <span
                              className={`abnormal-indicator ${explanation.status?.toLowerCase() === 'high' ? 'indicator-high' : 'indicator-low'}`}
                              onMouseEnter={(e) => handleMouseEnter(e, explanation, 'explanation')}
                              onMouseLeave={handleMouseLeave}
                            >
                              {explanation.status}
                            </span>
                          )}
                        </div>
                      ) : (
                        <span className="cell-value" title={String(cell)}>{cell}</span>
                      )}
                    </div>
                  ))}
                </div>
              )
            })}
          </div>
        </div>
      </GlassCard>

      {/* Tooltip Portal - rendered outside the card */}
      {activeTooltip && (
        <div
          className={`tooltip-portal ${tooltipPosition.side === 'left' ? 'tooltip-left' : 'tooltip-right'}`}
          style={{
            position: 'fixed',
            left: tooltipPosition.side === 'left' ? 'auto' : tooltipPosition.x,
            right: tooltipPosition.side === 'left' ? (window.innerWidth - tooltipPosition.x) : 'auto',
            top: tooltipPosition.y,
            transform: 'translateY(-50%)',
            zIndex: 9999
          }}
        >
          <div className="tooltip-content-visible">
            {tooltipType === 'explanation' ? (
              <>
                <div className="tooltip-header">
                  <span className={`tooltip-status ${activeTooltip.status?.toLowerCase() === 'high' ? 'status-high' : 'status-low'}`}>
                    {activeTooltip.status}
                  </span>
                  <span className="tooltip-test">{activeTooltip.test_name}</span>
                </div>
                {activeTooltip.value && (
                  <div className="tooltip-value">
                    Value: {activeTooltip.value}
                    {activeTooltip.reference_range && ` (Ref: ${activeTooltip.reference_range})`}
                  </div>
                )}
                <p className="tooltip-explanation">{activeTooltip.explanation}</p>
              </>
            ) : (
              <>
                <div className="tooltip-header">
                  <span className="tooltip-test">Report Details</span>
                </div>
                <div className="metadata-list">
                  {activeTooltip.lab_name && activeTooltip.lab_name !== 'null' && (
                    <div className="metadata-item">
                      <span className="metadata-label">Lab</span>
                      <span className="metadata-value">{activeTooltip.lab_name}</span>
                    </div>
                  )}
                  {activeTooltip.report_date && activeTooltip.report_date !== 'null' && (
                    <div className="metadata-item">
                      <span className="metadata-label">Date</span>
                      <span className="metadata-value">{activeTooltip.report_date}</span>
                    </div>
                  )}
                  {activeTooltip.sample_type && activeTooltip.sample_type !== 'null' && (
                    <div className="metadata-item">
                      <span className="metadata-label">Sample</span>
                      <span className="metadata-value">{activeTooltip.sample_type}</span>
                    </div>
                  )}
                  {activeTooltip.collection_time && activeTooltip.collection_time !== 'null' && (
                    <div className="metadata-item">
                      <span className="metadata-label">Collection</span>
                      <span className="metadata-value">{activeTooltip.collection_time}</span>
                    </div>
                  )}
                  {activeTooltip.patient_id && activeTooltip.patient_id !== 'null' && (
                    <div className="metadata-item">
                      <span className="metadata-label">Patient ID</span>
                      <span className="metadata-value">{activeTooltip.patient_id}</span>
                    </div>
                  )}
                  {activeTooltip.doctor_name && activeTooltip.doctor_name !== 'null' && (
                    <div className="metadata-item">
                      <span className="metadata-label">Doctor</span>
                      <span className="metadata-value">{activeTooltip.doctor_name}</span>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </>
  )
}

// ============ IMAGE LIGHTBOX WITH BOUNDING BOX ============

function ImageLightbox({ image, aiSummary, onClose }) {
  const canvasRef = useRef(null)
  const imageRef = useRef(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [boundingBox, setBoundingBox] = useState(null)
  const [startPos, setStartPos] = useState({ x: 0, y: 0 })
  const [question, setQuestion] = useState('')
  const [response, setResponse] = useState('')
  const [loading, setLoading] = useState(false)
  const [imageLoaded, setImageLoaded] = useState(false)

  const imageSrc = image?.image_base64 || (image?.image_path ? `file://${image.image_path}` : null)

  // Resize canvas to match image
  useEffect(() => {
    if (imageLoaded && imageRef.current && canvasRef.current) {
      const img = imageRef.current
      canvasRef.current.width = img.clientWidth
      canvasRef.current.height = img.clientHeight
    }
  }, [imageLoaded])

  // Draw the bounding box on canvas
  useEffect(() => {
    if (!canvasRef.current || !boundingBox) return
    const ctx = canvasRef.current.getContext('2d')
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)

    // Draw semi-transparent fill
    ctx.fillStyle = 'rgba(59, 130, 246, 0.25)'
    ctx.fillRect(boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height)

    // Draw border
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.9)'
    ctx.lineWidth = 3
    ctx.setLineDash([8, 4])
    ctx.strokeRect(boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height)

    // Draw corner markers
    ctx.setLineDash([])
    const cornerSize = 12
    ctx.lineWidth = 4
    const corners = [
      [boundingBox.x, boundingBox.y],
      [boundingBox.x + boundingBox.width, boundingBox.y],
      [boundingBox.x, boundingBox.y + boundingBox.height],
      [boundingBox.x + boundingBox.width, boundingBox.y + boundingBox.height]
    ]
    corners.forEach(([cx, cy]) => {
      ctx.beginPath()
      ctx.arc(cx, cy, 5, 0, 2 * Math.PI)
      ctx.fillStyle = 'rgba(59, 130, 246, 1)'
      ctx.fill()
    })
  }, [boundingBox])

  const handleMouseDown = (e) => {
    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    setStartPos({ x, y })
    setIsDrawing(true)
    setBoundingBox(null)
    setResponse('')
  }

  const handleMouseMove = (e) => {
    if (!isDrawing) return
    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    setBoundingBox({
      x: Math.min(startPos.x, x),
      y: Math.min(startPos.y, y),
      width: Math.abs(x - startPos.x),
      height: Math.abs(y - startPos.y)
    })
  }

  const handleMouseUp = () => {
    setIsDrawing(false)
  }

  const handleSubmitQuestion = async () => {
    if (!question.trim() || !boundingBox || !imageRef.current) return

    setLoading(true)
    setResponse('')

    try {
      const img = imageRef.current

      // Calculate scale factors for the backend
      const scaleX = img.naturalWidth / img.clientWidth
      const scaleY = img.naturalHeight / img.clientHeight

      // Get the original image as blob
      const imageBlob = await fetch(imageSrc).then(r => r.blob())

      const formData = new FormData()
      formData.append('image', imageBlob, 'image.png')
      formData.append('bbox_x', boundingBox.x.toString())
      formData.append('bbox_y', boundingBox.y.toString())
      formData.append('bbox_width', boundingBox.width.toString())
      formData.append('bbox_height', boundingBox.height.toString())
      formData.append('scale_x', scaleX.toString())
      formData.append('scale_y', scaleY.toString())
      formData.append('question', question)
      formData.append('context', aiSummary || '')
      formData.append('max_tokens', '2048')

      const res = await fetch(`${API_BASE_URL}/query-image-region`, {
        method: 'POST',
        body: formData
      })

      const data = await res.json()

      if (data.success && data.response) {
        // Handle OpenAI-style response format: choices[0].message.content
        let responseText = ''
        if (typeof data.response === 'string') {
          responseText = data.response
        } else if (data.response.choices?.[0]?.message?.content) {
          responseText = data.response.choices[0].message.content
        } else if (data.response.text) {
          responseText = data.response.text
        } else if (data.response.content) {
          responseText = data.response.content
        } else {
          responseText = JSON.stringify(data.response)
        }
        setResponse(responseText)
      } else {
        setResponse(`Error: ${data.error || 'Failed to get response'}`)
      }
    } catch (err) {
      setResponse(`Error: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const clearSelection = () => {
    setBoundingBox(null)
    setResponse('')
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d')
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
    }
  }

  if (!imageSrc) return null

  return (
    <div className="lightbox-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="lightbox-container">
        <button className="lightbox-close" onClick={onClose}>
          <X className="w-6 h-6" />
        </button>

        <div className="lightbox-content">
          <div className="lightbox-image-section">
            <div className="lightbox-image-container">
              <img
                ref={imageRef}
                src={imageSrc}
                alt={image.caption || 'Medical Image'}
                className="lightbox-image"
                onLoad={() => setImageLoaded(true)}
                draggable={false}
              />
              <canvas
                ref={canvasRef}
                className="lightbox-canvas"
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
              />
            </div>

            <div className="lightbox-instructions">
              <ZoomIn className="w-4 h-4" />
              <span>Click and drag to select a region, then ask a question about it</span>
            </div>
          </div>

          <div className="lightbox-question-section">
            <div className="lightbox-question-header">
              <h3>Ask About Selected Region</h3>
              {boundingBox && (
                <button className="clear-selection-btn" onClick={clearSelection}>
                  Clear Selection
                </button>
              )}
            </div>

            <div className="lightbox-input-wrapper">
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSubmitQuestion()}
                placeholder={boundingBox ? "Ask about the selected region..." : "Draw a box on the image first..."}
                className="lightbox-input"
                disabled={!boundingBox || loading}
              />
              <button
                className="lightbox-submit"
                onClick={handleSubmitQuestion}
                disabled={!boundingBox || !question.trim() || loading}
              >
                {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
              </button>
            </div>

            {response && (
              <div className="lightbox-response">
                <div className="response-header">AI Analysis</div>
                <div className="response-content">
                  {response.split('\n').map((para, i) => (
                    <p key={i}>{formatTextWithBold(para)}</p>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

// ============ IMAGE GALLERY ============

function ImageGallery({ images, detailedSummary, onImageClick }) {
  const containerRef = useRef(null)

  const hasImages = images && images.length > 0
  const hasContent = hasImages || detailedSummary

  if (!hasContent) return null

  const scroll = (dir) => {
    containerRef.current?.scrollBy({ left: dir * 300, behavior: 'smooth' })
  }

  return (
    <section className="gallery-section">
      <div className="gallery-header">
        <div>
          <h3 className="gallery-title">Scans & Graphics</h3>
          <p className="gallery-subtitle">Extracted Visual Media</p>
        </div>
        <div className="gallery-nav">
          <button onClick={() => scroll(-1)} className="gallery-btn">
            <ArrowLeft className="w-4 h-4" />
          </button>
          <button onClick={() => scroll(1)} className="gallery-btn">
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div ref={containerRef} className="gallery-container">
        {hasImages ? (
          images.map((img, i) => (
            <div
              key={i}
              className="gallery-item gallery-item-clickable"
              onClick={() => onImageClick && onImageClick(img)}
            >
              <div className="gallery-image-wrapper">
                {img.image_base64 ? (
                  <img
                    src={img.image_base64}
                    alt={img.caption || `Medical Image ${i + 1}`}
                    className="gallery-image"
                    onError={(e) => { e.target.style.display = 'none' }}
                  />
                ) : img.image_path ? (
                  <img
                    src={`file://${img.image_path}`}
                    alt={img.caption || `Medical Image ${i + 1}`}
                    className="gallery-image"
                    onError={(e) => { e.target.style.display = 'none' }}
                  />
                ) : (
                  <div className="gallery-placeholder">
                    <span>Image {i + 1}</span>
                  </div>
                )}
                <div className="gallery-zoom-overlay">
                  <ZoomIn className="w-6 h-6" />
                  <span>Click to zoom & annotate</span>
                </div>
              </div>
              <div className="gallery-meta">
                <span className="gallery-label">{img.caption || `Figure ${i + 1}`}</span>
                <Expand className="w-3 h-3 text-slate-400" />
              </div>
              {img.description && (
                <p className="gallery-desc">{img.description}</p>
              )}
            </div>
          ))
        ) : (
          <div className="gallery-empty">
            <p>No medical images detected in this report</p>
          </div>
        )}
      </div>
    </section>
  )
}

// ============ AI SUMMARY SECTION ============

function AISummary({ summary, delay }) {
  if (!summary) return null

  // Clean double asterisks and headers, but preserve single asterisks for bold
  const cleanSummary = summary
    .replace(/\*\*/g, '*')  // Convert ** to * for consistent bold formatting
    .replace(/#{1,6}\s/g, '')

  return (
    <section className="ai-summary-section">
      <GlassCard label="AI Analysis Summary" span="col-span-3" featured delay={delay}>
        <div className="summary-content">
          {cleanSummary.split('\n\n').map((para, i) => (
            <p key={i} className="summary-para">{formatTextWithBold(para)}</p>
          ))}
        </div>
      </GlassCard>
    </section>
  )
}

// ============ THREE.JS BACKGROUND ============

function ThreeBackground() {
  const containerRef = useRef(null)

  useEffect(() => {
    if (!containerRef.current) return

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000)
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true })

    renderer.setSize(window.innerWidth, window.innerHeight)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    containerRef.current.appendChild(renderer.domElement)

    const group = new THREE.Group()
    scene.add(group)

    const count = 180
    const geo = new THREE.BufferGeometry()
    const pos = []
    const cols = []

    for (let i = 0; i < count; i++) {
      const t = i / count
      const angle = t * Math.PI * 8
      const y = (t - 0.5) * 16
      const r = 4.5
      pos.push(Math.cos(angle) * r, y, Math.sin(angle) * r)
      cols.push(0.3, 0.4, 0.6)
      pos.push(Math.cos(angle + Math.PI) * r, y, Math.sin(angle + Math.PI) * r)
      cols.push(0.6, 0.7, 0.8)
    }

    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
    geo.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3))

    const mat = new THREE.PointsMaterial({
      size: 0.12,
      vertexColors: true,
      transparent: true,
      opacity: 0.7
    })

    const points = new THREE.Points(geo, mat)
    group.add(points)

    camera.position.z = 13

    const animate = () => {
      requestAnimationFrame(animate)
      group.rotation.y += 0.0015
      renderer.render(scene, camera)
    }
    animate()

    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight
      camera.updateProjectionMatrix()
      renderer.setSize(window.innerWidth, window.innerHeight)
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      containerRef.current?.removeChild(renderer.domElement)
      renderer.dispose()
    }
  }, [])

  return <div ref={containerRef} className="canvas-container" />
}

// ============ MAIN APP ============

function App() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [files, setFiles] = useState([])
  const [dragActive, setDragActive] = useState(false)

  // Chat state
  const [chatMessages, setChatMessages] = useState([])
  const [chatInput, setChatInput] = useState('')
  const [chatLoading, setChatLoading] = useState(false)
  const [chatPanelOpen, setChatPanelOpen] = useState(false)
  const [chatPanelWidth, setChatPanelWidth] = useState(380)
  const [isResizing, setIsResizing] = useState(false)

  // Lightbox state
  const [lightboxImage, setLightboxImage] = useState(null)
  const chatContainerRef = useRef(null)

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const newFiles = Array.from(e.dataTransfer.files)
      setFiles(prev => {
        // Combine and dedupe by filename
        const existing = new Set(prev.map(f => f.name))
        const unique = newFiles.filter(f => !existing.has(f.name))
        return [...prev, ...unique]
      })
    }
  }

  const handleFileChange = (e) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files)
      setFiles(prev => {
        // Combine and dedupe by filename
        const existing = new Set(prev.map(f => f.name))
        const unique = newFiles.filter(f => !existing.has(f.name))
        return [...prev, ...unique]
      })
    }
  }

  const removeFile = (index) => {
    setFiles(prev => prev.filter((_, i) => i !== index))
  }

  const handleSubmit = async () => {
    if (files.length === 0) return

    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    files.forEach(file => formData.append('files', file))

    try {
      const response = await fetch(`${API_BASE_URL}/process-pdf`, {
        method: 'POST',
        body: formData
      })
      const data = await response.json()

      if (data.success) {
        setResult(data)
      } else {
        setError(data.error || 'Processing failed')
      }
    } catch (err) {
      setError(`Connection failed: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  // Chat submission handler
  const handleChatSubmit = async () => {
    if (!chatInput.trim() || chatLoading) return

    const userMessage = chatInput.trim()
    setChatInput('')

    // Add user message to chat
    setChatMessages(prev => [...prev, { role: 'user', content: userMessage }])
    setChatLoading(true)

    try {
      // Build prompt with AI summary context
      const aiSummary = imageGallery?.detailed_summary || ''
      const contextPrompt = aiSummary
        ? `I have analyzed a medical report. Here is the AI analysis summary:\n\n${aiSummary}\n\nBased on this context, please answer the following question from the patient:\n\n${userMessage}`
        : userMessage

      const response = await fetch(`${API_BASE_URL}/predict-text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: contextPrompt,
          system_prompt: 'You are a helpful medical assistant. Provide clear, accurate, and compassionate responses based on the medical report analysis. Always recommend consulting with a healthcare provider for personalized medical advice.',
          max_tokens: 1024
        })
      })

      const data = await response.json()

      if (data.success && data.response) {
        // Handle OpenAI-style response format: choices[0].message.content
        let responseText = ''
        if (typeof data.response === 'string') {
          responseText = data.response
        } else if (data.response.choices?.[0]?.message?.content) {
          responseText = data.response.choices[0].message.content
        } else if (data.response.text) {
          responseText = data.response.text
        } else if (data.response.content) {
          responseText = data.response.content
        } else {
          responseText = JSON.stringify(data.response)
        }
        setChatMessages(prev => [...prev, { role: 'assistant', content: responseText }])
      } else {
        setChatMessages(prev => [...prev, { role: 'assistant', content: `Sorry, I couldn't process your question. ${data.error || ''}` }])
      }
    } catch (err) {
      setChatMessages(prev => [...prev, { role: 'assistant', content: `Error: ${err.message}` }])
    } finally {
      setChatLoading(false)
    }
  }

  // Scroll chat to bottom when new messages arrive
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight
    }
  }, [chatMessages])

  // Handle panel resize
  const handleResizeStart = useCallback((e) => {
    e.preventDefault()
    setIsResizing(true)
  }, [])

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isResizing) return
      const newWidth = window.innerWidth - e.clientX
      setChatPanelWidth(Math.max(280, Math.min(600, newWidth)))
    }

    const handleMouseUp = () => {
      setIsResizing(false)
    }

    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = 'ew-resize'
      document.body.style.userSelect = 'none'
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
  }, [isResizing])

  const hasResult = result && result.success

  // Extract data from result
  const reportData = result?.report_data || {}
  const tabularReports = result?.tabular_reports || []
  const imageGallery = result?.image_gallery || {}
  const llmSummary = result?.llm_summary || {}
  const aiSummary = imageGallery?.detailed_summary || ''

  return (
    <div className="app">
      {/* Ambient Background */}
      <div className="ambient-light">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
      </div>
      <div className="noise-overlay" />
      <ThreeBackground />

      {/* Header */}
      <header className="header">
        <div className="header-logo">
          <h1>MARS</h1>
        </div>
        <div className="header-info">
          <div className="header-meta">
            <div className="meta-label">Analysis Mode</div>
            <div className="meta-value">{hasResult ? 'Viewer' : 'Upload'}</div>
          </div>
        </div>
      </header>

      <main className="main-content">
        {!hasResult ? (
          /* Upload Section */
          <section className="upload-section">
            <div className="upload-hero">
              <div className="hero-status">
                <span className="ping-dot" />
                <span className="status-text">Ready for Analysis</span>
              </div>
              <h2 className="upload-title">Upload Medical Document</h2>
              <p className="upload-subtitle">Drop your PDF files below for AI-powered extraction and analysis</p>
            </div>

            <div
              className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <input
                type="file"
                multiple
                accept=".pdf"
                onChange={handleFileChange}
                className="file-input"
                id="file-input"
              />
              <label htmlFor="file-input" className="upload-label">
                <Upload className="upload-icon" />
                <span className="upload-text">
                  {files.length > 0 ? `${files.length} file(s) selected` : 'Click or drag files here'}
                </span>
                {files.length > 0 && (
                  <div className="file-tags">
                    {files.map((f, i) => (
                      <span key={i} className="data-tag file-tag">
                        {f.name}
                        <button
                          type="button"
                          className="file-remove-btn"
                          onClick={(e) => {
                            e.preventDefault()
                            e.stopPropagation()
                            removeFile(i)
                          }}
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </span>
                    ))}
                  </div>
                )}
              </label>
            </div>

            <button
              onClick={handleSubmit}
              disabled={loading || files.length === 0}
              className="submit-btn"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <span>Analyze Documents</span>
                  <ArrowRight className="w-5 h-5" />
                </>
              )}
            </button>

            {error && (
              <div className="error-box">
                <p>{error}</p>
              </div>
            )}
          </section>
        ) : (
          /* Results Dashboard */
          <>
            {/* Hero with Patient Name */}
            <PatientHero
              patientInfo={reportData.patient_info}
              reportDate={reportData.report_date}
            />

            {/* Main Dashboard Grid */}
            <div className="dashboard-grid">
              {/* Diagnosis - Featured */}
              <DiagnosisCard
                diagnosis={reportData.report_summary?.diagnosis}
                delay={0}
              />

              {/* Doctor Info */}
              <KeyValueCard
                label="doctor_info"
                data={reportData.doctor_info}
                delay={1}
              />

              {/* Main Findings */}
              <TextCard
                label="main_findings"
                content={reportData.report_summary?.main_findings}
                delay={2}
              />

              {/* Patient Explanation */}
              <TextCard
                label="patient_explanation"
                content={reportData.report_summary?.patient_explanation}
                delay={3}
                span="col-span-2"
              />

              {/* Recommendations */}
              <TextCard
                label="recommendations"
                content={reportData.report_summary?.recommendations}
                delay={4}
              />

              {/* Medications */}
              <MedicationsCard
                medications={reportData.medications}
                delay={5}
              />

              {/* Next Appointment */}
              {reportData.next_appointment && (
                <GlassCard label="Forward Planning" delay={6}>
                  <p className="text-content">{reportData.next_appointment}</p>
                </GlassCard>
              )}

              {/* Patient History */}
              {reportData.patient_history && (
                <TextCard
                  label="patient_history"
                  content={reportData.patient_history}
                  delay={7}
                />
              )}

              {/* Tabular Reports */}
              {tabularReports.flatMap((report, ri) =>
                (report.tables || []).map((table, ti) => (
                  <DataTableCard
                    key={`${ri}-${ti}`}
                    tableData={table}
                    reportMetadata={report.metadata}
                    delay={8 + ri * 2 + ti}
                  />
                ))
              )}
            </div>

            {/* Image Gallery */}
            <ImageGallery
              images={imageGallery.medical_images}
              detailedSummary={imageGallery.detailed_summary}
              onImageClick={(img) => setLightboxImage(img)}
            />

            {/* AI Summary */}
            <AISummary
              summary={imageGallery.detailed_summary}
              delay={12}
            />

            {/* Reset Button */}
            <button
              onClick={() => { setResult(null); setFiles([]); setChatMessages([]) }}
              className="reset-btn"
            >
              ← Upload new document
            </button>
          </>
        )}
      </main>

      {/* Chat Toggle Button - Only show when results are available */}
      {hasResult && !chatPanelOpen && (
        <button
          className="chat-toggle-btn"
          onClick={() => setChatPanelOpen(true)}
        >
          <MessageSquarePlus className="w-5 h-5" />
          <span>Ask Questions</span>
        </button>
      )}

      {/* Chat Side Panel */}
      {hasResult && chatPanelOpen && (
        <div
          className={`chat-panel ${isResizing ? 'chat-panel-resizing' : ''}`}
          style={{ width: chatPanelWidth }}
        >
          {/* Resize Handle */}
          <div
            className="chat-panel-resize-handle"
            onMouseDown={handleResizeStart}
          />

          {/* Panel Header */}
          <div className="chat-panel-header">
            <div className="chat-panel-title">
              <MessageSquarePlus className="w-4 h-4" />
              <span>Ask About Results</span>
            </div>
            <button
              className="chat-panel-close"
              onClick={() => setChatPanelOpen(false)}
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Chat Messages */}
          <div className="chat-panel-messages" ref={chatContainerRef}>
            {chatMessages.length === 0 ? (
              <div className="chat-empty-state">
                <MessageSquarePlus className="w-8 h-8" />
                <p>Ask questions about your medical report</p>
                <span>The AI will use the analysis context to provide informed answers</span>
              </div>
            ) : (
              <>
                {chatMessages.map((msg, i) => (
                  <div key={i} className={`chat-message ${msg.role === 'user' ? 'chat-message-user' : 'chat-message-assistant'}`}>
                    <div className="message-role">{msg.role === 'user' ? 'You' : 'AI Assistant'}</div>
                    <div className="message-content">
                      {msg.content.split('\n').map((para, j) => (
                        <p key={j}>{formatTextWithBold(para)}</p>
                      ))}
                    </div>
                  </div>
                ))}
                {chatLoading && (
                  <div className="chat-message chat-message-assistant">
                    <div className="message-role">AI Assistant</div>
                    <div className="message-content message-loading">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Thinking...</span>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          {/* Chat Input */}
          <div className="chat-panel-input">
            <input
              type="text"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleChatSubmit()}
              placeholder="Type your question..."
              className="chat-input"
              disabled={chatLoading}
            />
            <button
              className="chat-submit"
              onClick={handleChatSubmit}
              disabled={chatLoading || !chatInput.trim()}
            >
              {chatLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            </button>
          </div>
        </div>
      )}

      {/* Image Lightbox */}
      {lightboxImage && (
        <ImageLightbox
          image={lightboxImage}
          aiSummary={aiSummary}
          onClose={() => setLightboxImage(null)}
        />
      )}
    </div>
  )
}

export default App
