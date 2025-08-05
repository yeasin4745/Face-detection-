import { useState, useRef, useEffect } from 'react'
import * as tf from '@tensorflow/tfjs'
import * as cocoSsd from '@tensorflow-models/coco-ssd'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Camera, CameraOff, Loader2, AlertCircle } from 'lucide-react'
import './App.css'

function App() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [model, setModel] = useState(null)
  const [isModelLoading, setIsModelLoading] = useState(false)
  const [isCameraActive, setIsCameraActive] = useState(false)
  const [detections, setDetections] = useState([])
  const [error, setError] = useState(null)
  const [stream, setStream] = useState(null)

  // Load the COCO-SSD model
  const loadModel = async () => {
    setIsModelLoading(true)
    setError(null)
    try {
      console.log('Loading TensorFlow.js...')
      await tf.ready()
      console.log('Loading COCO-SSD model...')
      const loadedModel = await cocoSsd.load()
      setModel(loadedModel)
      console.log('Model loaded successfully!')
    } catch (err) {
      console.error('Error loading model:', err)
      setError('Failed to load AI model. Please refresh and try again.')
    } finally {
      setIsModelLoading(false)
    }
  }

  // Start webcam
  const startCamera = async () => {
    setError(null)
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      })
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
        videoRef.current.play()
        setStream(mediaStream)
        setIsCameraActive(true)
      }
    } catch (err) {
      console.error('Error accessing camera:', err)
      setError('Failed to access camera. Please ensure camera permissions are granted.')
    }
  }

  // Stop webcam
  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop())
      setStream(null)
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setIsCameraActive(false)
    setDetections([])
  }

  // Detect objects in the video frame
  const detectObjects = async () => {
    if (model && videoRef.current && canvasRef.current && isCameraActive) {
      const video = videoRef.current
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')

      // Set canvas dimensions to match video
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      try {
        // Perform detection
        const predictions = await model.detect(video)
        setDetections(predictions)

        // Draw bounding boxes
        predictions.forEach(prediction => {
          const [x, y, width, height] = prediction.bbox
          const confidence = (prediction.score * 100).toFixed(1)

          // Draw bounding box
          ctx.strokeStyle = '#00ff00'
          ctx.lineWidth = 2
          ctx.strokeRect(x, y, width, height)

          // Draw label background
          ctx.fillStyle = '#00ff00'
          ctx.fillRect(x, y - 25, width, 25)

          // Draw label text
          ctx.fillStyle = '#000000'
          ctx.font = '16px Arial'
          ctx.fillText(`${prediction.class} (${confidence}%)`, x + 5, y - 5)
        })
      } catch (err) {
        console.error('Error during detection:', err)
      }
    }
  }

  // Run detection loop
  useEffect(() => {
    let animationId
    
    const runDetection = () => {
      detectObjects()
      animationId = requestAnimationFrame(runDetection)
    }

    if (model && isCameraActive) {
      runDetection()
    }

    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId)
      }
    }
  }, [model, isCameraActive])

  // Load model on component mount
  useEffect(() => {
    loadModel()
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            AI Object Detection
          </h1>
          <p className="text-lg text-gray-600">
            Real-time object detection using your webcam and TensorFlow.js
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Video Feed */}
          <div className="lg:col-span-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Camera className="w-5 h-5" />
                  Camera Feed
                </CardTitle>
                <CardDescription>
                  Live video feed with object detection overlay
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="relative bg-black rounded-lg overflow-hidden">
                  <video
                    ref={videoRef}
                    className="w-full h-auto"
                    autoPlay
                    muted
                    playsInline
                  />
                  {!isCameraActive && (
                    <div className="absolute inset-0 flex items-center justify-center bg-gray-800">
                      <div className="text-center text-white">
                        <CameraOff className="w-16 h-16 mx-auto mb-4 opacity-50" />
                        <p>Camera not active</p>
                      </div>
                    </div>
                  )}
                  <canvas
                    ref={canvasRef}
                    className="absolute top-0 left-0 w-full h-full"
                  />
                </div>

                <div className="flex gap-2 mt-4">
                  <Button
                    onClick={isCameraActive ? stopCamera : startCamera}
                    disabled={isModelLoading}
                    variant={isCameraActive ? "destructive" : "default"}
                  >
                    {isCameraActive ? (
                      <>
                        <CameraOff className="w-4 h-4 mr-2" />
                        Stop Camera
                      </>
                    ) : (
                      <>
                        <Camera className="w-4 h-4 mr-2" />
                        Start Camera
                      </>
                    )}
                  </Button>

                  {!model && (
                    <Button
                      onClick={loadModel}
                      disabled={isModelLoading}
                      variant="outline"
                    >
                      {isModelLoading ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Loading Model...
                        </>
                      ) : (
                        'Reload Model'
                      )}
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Detection Results */}
          <div>
            <Card>
              <CardHeader>
                <CardTitle>Detection Results</CardTitle>
                <CardDescription>
                  Objects detected in real-time
                </CardDescription>
              </CardHeader>
              <CardContent>
                {/* Model Status */}
                <div className="mb-4">
                  <h3 className="font-semibold mb-2">Model Status</h3>
                  {isModelLoading ? (
                    <Badge variant="secondary">
                      <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                      Loading...
                    </Badge>
                  ) : model ? (
                    <Badge variant="default">Ready</Badge>
                  ) : (
                    <Badge variant="destructive">Not Loaded</Badge>
                  )}
                </div>

                {/* Error Display */}
                {error && (
                  <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                    <div className="flex items-center gap-2 text-red-700">
                      <AlertCircle className="w-4 h-4" />
                      <span className="text-sm">{error}</span>
                    </div>
                  </div>
                )}

                {/* Detections List */}
                <div>
                  <h3 className="font-semibold mb-2">
                    Detected Objects ({detections.length})
                  </h3>
                  {detections.length > 0 ? (
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                      {detections.map((detection, index) => (
                        <div
                          key={index}
                          className="flex justify-between items-center p-2 bg-gray-50 rounded"
                        >
                          <span className="font-medium capitalize">
                            {detection.class}
                          </span>
                          <Badge variant="outline">
                            {(detection.score * 100).toFixed(1)}%
                          </Badge>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-gray-500 text-sm">
                      {isCameraActive ? 'No objects detected' : 'Start camera to begin detection'}
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Instructions */}
            <Card className="mt-4">
              <CardHeader>
                <CardTitle>Instructions</CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-gray-600 space-y-2">
                <p>1. Click "Start Camera" to begin</p>
                <p>2. Allow camera permissions when prompted</p>
                <p>3. Point camera at objects to detect them</p>
                <p>4. Green boxes will appear around detected objects</p>
                <p>5. Detection results appear in real-time</p>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-8 text-gray-500 text-sm">
          <p>Powered by TensorFlow.js and COCO-SSD model</p>
          <p>Detects 80+ common objects including people, animals, vehicles, and household items</p>
        </div>
      </div>
    </div>
  )
}

export default App
      
