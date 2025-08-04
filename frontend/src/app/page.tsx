"use client"

import type React from "react"
import { useState, useEffect, useRef, useCallback, useMemo } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Upload, Send, Download, MessageCircle, ImageIcon, Volume2, User, LogOut, AlertCircle } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { useFormInput } from "@/hooks/use-form-input"
import { Progress } from "@/components/ui/progress"

type Page = "chatbot" | "photo-editor" | "text-to-speech" | "file-upload" | "login" | "signup"
type AppUser = { username: string; email: string } | null

interface Activity {
  id: string
  timestamp: string
  activity_type: string
  details: any
  output_url?: string
}

// Add this interface near the top with other interfaces
interface TTSResponse {
  audio_url: string;
  background_music: {
    name: string;
    duration: number;
    audio_url: string;
  };
  filename: string;
  bg_filename: string;
  has_background_music: boolean;
  selected_track: string;
}

// Simple form state management without external hook
function useSimpleForm<T>(initialValues: T) {
  const [values, setValues] = useState<T>(initialValues)

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setValues((prev) => ({ ...prev, [name]: value }))
  }, [])

  const reset = useCallback(() => {
    setValues(initialValues)
  }, [initialValues])

  return { values, handleChange, reset }
}

export default function AIToolsHub() {
  const [currentPage, setCurrentPage] = useState<Page>("login")
  const [user, setUser] = useState<AppUser>(null)
  const [activities, setActivities] = useState<Activity[]>([])
  const [loading, setLoading] = useState(false)
  const [authToken, setAuthToken] = useState<string | null>(null)

  // Photo Editor state
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [selectedTransform, setSelectedTransform] = useState("")
  const [transformedImage, setTransformedImage] = useState<string | null>(null)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [imageUploadProgress, setImageUploadProgress] = useState(0)
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Text-to-Speech state - Fixed to prevent re-renders
  const ttsForm = useFormInput({
    text: "",
    music: "",
  })
  const [selectedVoice, setSelectedVoice] = useState("")
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [availableVoices, setAvailableVoices] = useState<Array<{ name: string; display_name: string }>>([])
  const [audioProgress, setAudioProgress] = useState(0)
  const [audioDuration, setAudioDuration] = useState(0)
  const audioRef = useRef<HTMLAudioElement>(null)
  const [audioKey, setAudioKey] = useState(0) // Key to force audio element recreation when needed

  // Auth state
  const loginForm = useFormInput({ username: "", password: "" })
  const signupForm = useFormInput({
    email: "",
    username: "",
    password: "",
    confirmPassword: "",
  })
  const [authError, setAuthError] = useState("")

  // API Base URL
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000"

  // Memoized functions to prevent unnecessary re-renders
  const fetchActivities = useCallback(async () => {
    if (!user) return

    try {
      const response = await fetch(`${API_BASE}/api/activities`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${authToken}`,
        },
        body: JSON.stringify({
          activity_type: getActivityType(currentPage),
        }),
      })

      if (response.ok) {
        const data = await response.json()
        setActivities(data.activities || [])
      }
    } catch (error) {
      console.error("Error fetching activities:", error)
    }
  }, [user, currentPage, authToken])

  const getActivityType = useCallback((page: Page): string => {
    const typeMap: Record<Page, string> = {
      chatbot: "chat",
      "photo-editor": "photo_edit",
      "text-to-speech": "text_to_speech",
      "file-upload": "file_upload",
      login: "",
      signup: "",
    }
    return typeMap[page] || ""
  }, [])

  useEffect(() => {
    if (user && currentPage !== "login" && currentPage !== "signup") {
      fetchActivities()
    }
  }, [user, currentPage, fetchActivities])

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setAuthError("")

    try {
      const response = await fetch(`${API_BASE}/api/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(loginForm.values),
      })

      const data = await response.json()

      if (response.ok) {
        setAuthToken(data.access_token)
        setUser(data.user)
        localStorage.setItem("auth_token", data.access_token)
        setCurrentPage("photo-editor")
        loginForm.reset()
      } else {
        setAuthError(data.error || "Login failed")
      }
    } catch (error) {
      setAuthError("Connection error. Please try again.")
    } finally {
      setLoading(false)
    }
  }

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setAuthError("")

    if (signupForm.values.password !== signupForm.values.confirmPassword) {
      setAuthError("Passwords do not match")
      setLoading(false)
      return
    }

    try {
      const response = await fetch(`${API_BASE}/api/signup`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          email: signupForm.values.email,
          username: signupForm.values.username,
          password: signupForm.values.password,
        }),
      })

      const data = await response.json()

      if (response.ok) {
        setAuthToken(data.access_token)
        setUser(data.user)
        localStorage.setItem("auth_token", data.access_token)
        setCurrentPage("chatbot")
        signupForm.reset()
      } else {
        setAuthError(data.error || "Signup failed")
      }
    } catch (error) {
      setAuthError("Connection error. Please try again.")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    const token = localStorage.getItem("auth_token")
    if (token) {
      setAuthToken(token)
      fetch(`${API_BASE}/api/me`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      })
        .then((response) => response.json())
        .then((data) => {
          if (data.username) {
            setUser(data)
            setCurrentPage("chatbot")
          } else {
            localStorage.removeItem("auth_token")
          }
        })
        .catch(() => {
          localStorage.removeItem("auth_token")
        })
    }
  }, [])

  const handleLogout = () => {
    setUser(null)
    setCurrentPage("login")
    setActivities([])
    localStorage.removeItem("auth_token")
    setAuthToken(null)
  }

  // Image handling functions
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelection(e.dataTransfer.files[0])
    }
  }

  const handleFileSelection = (file: File) => {
    if (!file.type.startsWith("image/")) {
      alert("Please select an image file")
      return
    }

    if (file.size > 10 * 1024 * 1024) {
      alert("File size must be less than 10MB")
      return
    }

    const url = URL.createObjectURL(file)
    setUploadedImage(url)
    setUploadedFile(file)
    setTransformedImage(null)
  }

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      handleFileSelection(file)
    }
  }

  const triggerFileInput = () => {
    fileInputRef.current?.click()
  }

  // Fixed image transformation to match working Streamlit code
  const transformImage = async () => {
    if (!uploadedFile || !selectedTransform || !user) return

    setLoading(true)
    setImageUploadProgress(0)

    try {
      const formData = new FormData()
      formData.append("file", uploadedFile, uploadedFile.name)
      formData.append("transformation", selectedTransform)

      console.log("Uploading file:", uploadedFile.name, "Size:", uploadedFile.size, "Type:", uploadedFile.type)

      const progressInterval = setInterval(() => {
        setImageUploadProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return 90
          }
          return prev + 10
        })
      }, 200)

      const response = await fetch(`${API_BASE}/api/upload-image`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${authToken}`,
        },
        body: formData,
      })

      clearInterval(progressInterval)
      setImageUploadProgress(100)

      const data = await response.json()
      console.log("Upload response:", data)

      if (response.ok) {
        setTransformedImage(data.transformed_url)
        fetchActivities()
        console.log("Image transformation successful:", data.transformed_url)
      } else {
        console.error("Transform failed:", data)
        alert(`Transform failed: ${data.error || "Unknown error"}`)
      }
    } catch (error) {
      console.error("Transform error:", error)
      alert("Failed to transform image. Please try again.")
    } finally {
      setLoading(false)
      setTimeout(() => setImageUploadProgress(0), 1000)
    }
  }

  // Fixed Text-to-Speech functions to prevent audio interruption
  const [ttsResponse, setTtsResponse] = useState<TTSResponse | null>(null);

  const generateSpeech = useCallback(async () => {
    if (!ttsForm.values.text.trim() || !user) return;

    setLoading(true);
    const previousAudioUrl = audioUrl;

    try {
      const response = await fetch(`${API_BASE}/api/text-to-speech`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${authToken}`,
        },
        body: JSON.stringify({
          text: ttsForm.values.text,
          voice: selectedVoice,
          music_search: ttsForm.values.music,
        }),
      });

      const data = await response.json();
      setTtsResponse(data); // Store the response data

      if (response.ok) {
        if (previousAudioUrl && previousAudioUrl.startsWith("blob:")) {
          URL.revokeObjectURL(previousAudioUrl)
        }

        if (data.audio_url) {
          const fullAudioUrl = data.audio_url.startsWith("http") ? data.audio_url : `${API_BASE}${data.audio_url}`

          try {
            const audioResponse = await fetch(fullAudioUrl)
            if (audioResponse.ok) {
              const blob = await audioResponse.blob()
              const newAudioUrl = URL.createObjectURL(blob)

              setAudioUrl(newAudioUrl)
              setAudioBlob(blob)
              setIsPlaying(false)
              setAudioProgress(0)
              setAudioKey((prev) => prev + 1)

              console.log("Audio loaded successfully, size:", blob.size)
            } else {
              console.error("Failed to fetch audio:", audioResponse.status)
              createDemoAudio()
            }
          } catch (fetchError) {
            console.error("Error fetching audio:", fetchError)
            createDemoAudio()
          }
        } else {
          createDemoAudio()
        }
        fetchActivities()
      } else {
        console.error("TTS failed:", data.error)
        alert(`Speech generation failed: ${data.error || "Unknown error"}`)
      }
    } catch (error) {
      console.error("TTS error:", error)
      alert("Failed to generate speech. Please try again.")
    } finally {
      setLoading(false)
    }
  }, [ttsForm.values.text, selectedVoice, ttsForm.values.music, user, authToken, audioUrl, fetchActivities])

  const createDemoAudio = useCallback(() => {
    try {
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)()
      const sampleRate = audioContext.sampleRate
      const duration = Math.min(ttsForm.values.text.length * 0.1, 5)
      const frameCount = sampleRate * duration
      const arrayBuffer = audioContext.createBuffer(1, frameCount, sampleRate)
      const channelData = arrayBuffer.getChannelData(0)

      for (let i = 0; i < frameCount; i++) {
        channelData[i] = Math.sin((2 * Math.PI * 440 * i) / sampleRate) * 0.1
      }

      const wavBlob = audioBufferToWav(arrayBuffer)
      const newAudioUrl = URL.createObjectURL(wavBlob)

      setAudioUrl(newAudioUrl)
      setAudioBlob(wavBlob)
      setIsPlaying(false)
      setAudioProgress(0)
      setAudioKey((prev) => prev + 1)
    } catch (error) {
      console.error("Error creating demo audio:", error)
    }
  }, [ttsForm.values.text])

  const audioBufferToWav = (buffer: AudioBuffer): Blob => {
    const length = buffer.length
    const arrayBuffer = new ArrayBuffer(44 + length * 2)
    const view = new DataView(arrayBuffer)
    const channelData = buffer.getChannelData(0)

    const writeString = (offset: number, string: string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i))
      }
    }

    writeString(0, "RIFF")
    view.setUint32(4, 36 + length * 2, true)
    writeString(8, "WAVE")
    writeString(12, "fmt ")
    view.setUint32(16, 16, true)
    view.setUint16(20, 1, true)
    view.setUint16(22, 1, true)
    view.setUint32(24, buffer.sampleRate, true)
    view.setUint32(28, buffer.sampleRate * 2, true)
    view.setUint16(32, 2, true)
    view.setUint16(34, 16, true)
    writeString(36, "data")
    view.setUint32(40, length * 2, true)

    let offset = 44
    for (let i = 0; i < length; i++) {
      const sample = Math.max(-1, Math.min(1, channelData[i]))
      view.setInt16(offset, sample * 0x7fff, true)
      offset += 2
    }

    return new Blob([arrayBuffer], { type: "audio/wav" })
  }

  // Memoized audio control functions to prevent re-renders
  const togglePlayPause = useCallback(() => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause()
      } else {
        audioRef.current.play().catch((error) => {
          console.error("Error playing audio:", error)
          alert("Unable to play audio. Please try downloading it instead.")
        })
      }
    }
  }, [isPlaying])

  const handleAudioTimeUpdate = useCallback(() => {
    if (audioRef.current && audioRef.current.duration) {
      const progress = (audioRef.current.currentTime / audioRef.current.duration) * 100
      setAudioProgress(progress)
    }
  }, [])

  const handleAudioLoadedMetadata = useCallback(() => {
    if (audioRef.current) {
      setAudioDuration(audioRef.current.duration)
    }
  }, [])

  const handleAudioPlay = useCallback(() => {
    setIsPlaying(true)
  }, [])

  const handleAudioPause = useCallback(() => {
    setIsPlaying(false)
  }, [])

  const handleAudioEnded = useCallback(() => {
    setIsPlaying(false)
    setAudioProgress(0)
  }, [])

  const downloadAudio = useCallback(() => {
    if (audioBlob) {
      const url = URL.createObjectURL(audioBlob)
      const a = document.createElement("a")
      a.href = url
      a.download = `speech_${Date.now()}.wav`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } else if (audioUrl) {
      const a = document.createElement("a")
      a.href = audioUrl
      a.download = `speech_${Date.now()}.wav`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
    }
  }, [audioBlob, audioUrl])

  useEffect(() => {
    const fetchVoices = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/voices`)
        const data = await response.json()
        if (response.ok) {
          setAvailableVoices(data.voices || [])
          if (data.voices && data.voices.length > 0 && !selectedVoice) {
            setSelectedVoice(data.voices[0].name)
          }
        }
      } catch (error) {
        console.error("Error fetching voices:", error)
      }
    }

    if (authToken) {
      fetchVoices()
    }
  }, [authToken, selectedVoice])

  const Navigation = useCallback(
    () => (
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex space-x-8">
              <div className="flex items-center">
                <h1 className="text-xl font-bold text-gray-900">AI Tools Hub</h1>
              </div>
              {user && (
                <div className="flex space-x-4 items-center">
                  <Button
                    variant={currentPage === "photo-editor" ? "default" : "ghost"}
                    onClick={() => setCurrentPage("photo-editor")}
                    className="flex items-center gap-2"
                  >
                    <ImageIcon size={16} />
                    Photo Editor
                  </Button>
                  <Button
                    variant={currentPage === "text-to-speech" ? "default" : "ghost"}
                    onClick={() => setCurrentPage("text-to-speech")}
                    className="flex items-center gap-2"
                  >
                    <Volume2 size={16} />
                    Text to Speech
                  </Button>
                  <Button
                    variant={currentPage === "file-upload" ? "default" : "ghost"}
                    onClick={() => setCurrentPage("file-upload")}
                    className="flex items-center gap-2"
                  >
                    <Upload size={16} />
                    File Upload
                  </Button>
                </div>
              )}
            </div>
            <div className="flex items-center space-x-4">
              {user ? (
                <>
                  <div className="flex items-center gap-2 text-sm text-gray-700">
                    <User size={16} />
                    Welcome, {user.username}
                  </div>
                  <Button variant="outline" onClick={handleLogout} className="flex items-center gap-2">
                    <LogOut size={16} />
                    Logout
                  </Button>
                </>
              ) : (
                <>
                  <Button
                    variant={currentPage === "login" ? "default" : "ghost"}
                    onClick={() => setCurrentPage("login")}
                  >
                    Login
                  </Button>
                  <Button
                    variant={currentPage === "signup" ? "default" : "ghost"}
                    onClick={() => setCurrentPage("signup")}
                  >
                    Sign Up
                  </Button>
                </>
              )}
            </div>
          </div>
        </div>
      </nav>
    ),
    [currentPage, user, handleLogout],
  )

  const ActivityHistory = () => (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="text-lg">Recent Activity</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 max-h-96 overflow-y-auto">
        {activities.length > 0 ? (
          activities.map((activity, index) => (
            <div key={activity.id || index} className="p-3 bg-gray-50 rounded-lg">
              <div className="text-xs text-gray-500">{new Date(activity.timestamp).toLocaleString()}</div>
              <div className="font-medium text-sm capitalize">{activity.activity_type.replace("_", " ")}</div>
              <div className="text-sm text-gray-700">
                {activity.activity_type === "chat" && activity.details?.user_message && (
                  <div className="space-y-1">
                    <div className="font-medium">You:</div>
                    <div className="text-xs bg-blue-100 p-2 rounded">{activity.details.user_message}</div>
                    <div className="font-medium">Bot:</div>
                    <div className="text-xs bg-gray-100 p-2 rounded">{activity.details.bot_response}</div>
                  </div>
                )}
                {activity.activity_type === "photo_edit" && (
                  <div>
                    <div>Transform: {activity.details?.transformation}</div>
                    {activity.output_url && (
                      <img
                        src={activity.output_url || "/placeholder.svg"}
                        alt="Output"
                        className="w-full h-20 object-cover rounded mt-2"
                      />
                    )}
                  </div>
                )}
                {activity.activity_type === "text_to_speech" && (
                  <div>
                    <div>Text: {activity.details?.text?.substring(0, 50)}...</div>
                    <div>Voice: {activity.details?.voice}</div>
                  </div>
                )}
              </div>
            </div>
          ))
        ) : (
          <div className="text-gray-500 text-sm">No recent activity</div>
        )}
      </CardContent>
    </Card>
  )

  const LoginPage = useMemo(
    () => (
      <div className="flex items-center justify-center min-h-[calc(100vh-4rem)]">
        <Card className="w-full max-w-md">
          <CardHeader>
            <CardTitle className="text-center">Welcome Back!</CardTitle>
          </CardHeader>
          <CardContent>
            {authError && (
              <Alert className="mb-4">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{authError}</AlertDescription>
              </Alert>
            )}
            <form onSubmit={handleLogin} className="space-y-4">
              <Input
                name="username"
                placeholder="Username"
                value={loginForm.values.username}
                onChange={loginForm.handleChange}
                required
                autoComplete="username"
              />
              <Input
                name="password"
                type="password"
                placeholder="Password"
                value={loginForm.values.password}
                onChange={loginForm.handleChange}
                required
                autoComplete="current-password"
              />
              <Button type="submit" className="w-full" disabled={loading}>
                {loading ? "Logging in..." : "Login"}
              </Button>
            </form>
            <div className="text-center mt-4">
              <span className="text-sm text-gray-600">Don't have an account? </span>
              <Button variant="link" onClick={() => setCurrentPage("signup")} className="p-0">
                Sign Up
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    ),
    [authError, loginForm.values, handleLogin, loginForm.handleChange, loading],
  )

  const SignupPage = useMemo(
    () => (
      <div className="flex items-center justify-center min-h-[calc(100vh-4rem)]">
        <Card className="w-full max-w-md">
          <CardHeader>
            <CardTitle className="text-center">Create Account</CardTitle>
          </CardHeader>
          <CardContent>
            {authError && (
              <Alert className="mb-4">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{authError}</AlertDescription>
              </Alert>
            )}
            <form onSubmit={handleSignup} className="space-y-4">
              <Input
                name="email"
                type="email"
                placeholder="Email"
                value={signupForm.values.email}
                onChange={signupForm.handleChange}
                required
                autoComplete="email"
              />
              <Input
                name="username"
                placeholder="Username"
                value={signupForm.values.username}
                onChange={signupForm.handleChange}
                required
                autoComplete="username"
              />
              <Input
                name="password"
                type="password"
                placeholder="Password"
                value={signupForm.values.password}
                onChange={signupForm.handleChange}
                required
                autoComplete="new-password"
              />
              <Input
                name="confirmPassword"
                type="password"
                placeholder="Confirm Password"
                value={signupForm.values.confirmPassword}
                onChange={signupForm.handleChange}
                required
                autoComplete="new-password"
              />
              <Button type="submit" className="w-full" disabled={loading}>
                {loading ? "Creating Account..." : "Sign Up"}
              </Button>
            </form>
            <div className="text-center mt-4">
              <span className="text-sm text-gray-600">Already have an account? </span>
              <Button variant="link" onClick={() => setCurrentPage("login")} className="p-0">
                Login
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    ),
    [authError, signupForm.values, handleSignup, signupForm.handleChange, loading],
  )

  const PhotoEditorPage = useMemo(
    () => (
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[calc(100vh-8rem)]">
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Image Upload</CardTitle>
            </CardHeader>
            <CardContent>
              <div
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                  dragActive
                    ? "border-blue-500 bg-blue-50"
                    : uploadedImage
                      ? "border-green-500 bg-green-50"
                      : "border-gray-300 bg-gray-50"
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={triggerFileInput}
                style={{ cursor: "pointer" }}
              >
                {uploadedImage ? (
                  <div className="space-y-4">
                    <img
                      src={uploadedImage || "/placeholder.svg"}
                      alt="Uploaded"
                      className="max-w-full h-48 mx-auto object-contain rounded"
                    />
                    <p className="text-sm text-green-600">✓ Image uploaded successfully</p>
                    <Button variant="outline" onClick={triggerFileInput}>
                      Choose Different Image
                    </Button>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <Upload className="mx-auto h-12 w-12 text-gray-400" />
                    <div>
                      <p className="text-lg font-medium text-gray-700">
                        {dragActive ? "Drop image here" : "Click or drag image here"}
                      </p>
                      <p className="text-sm text-gray-500 mt-1">Supports: JPG, PNG, GIF (max 10MB)</p>
                    </div>
                  </div>
                )}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="hidden"
                />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Transformation Options</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Select value={selectedTransform} onValueChange={setSelectedTransform}>
                <SelectTrigger>
                  <SelectValue placeholder="Select transformation" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="blur">Blur Effect</SelectItem>
                  <SelectItem value="grayscale">Grayscale</SelectItem>
                  <SelectItem value="sepia">Sepia Tone</SelectItem>
                  <SelectItem value="flip-h">Flip Horizontal</SelectItem>
                  <SelectItem value="flip-v">Flip Vertical</SelectItem>
                </SelectContent>
              </Select>

              {imageUploadProgress > 0 && imageUploadProgress < 100 && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Processing...</span>
                    <span>{imageUploadProgress}%</span>
                  </div>
                  <Progress value={imageUploadProgress} className="w-full" />
                </div>
              )}

              <Button
                onClick={transformImage}
                disabled={!uploadedImage || !selectedTransform || loading}
                className="w-full"
              >
                {loading ? "Processing..." : "Apply Transformation"}
              </Button>
            </CardContent>
          </Card>

          {transformedImage && (
            <Card>
              <CardHeader>
                <CardTitle>Transformed Image</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <img
                  src={transformedImage || "/placeholder.svg"}
                  alt="Transformed"
                  className="max-w-full h-48 mx-auto object-contain rounded"
                />
                <div className="flex gap-2">
                  <Button
                    className="flex-1"
                    variant="outline"
                    onClick={() => {
                      const a = document.createElement("a")
                      a.href = transformedImage
                      a.download = `transformed_${Date.now()}.jpg`
                      a.click()
                    }}
                  >
                    <Download size={16} className="mr-2" />
                    Download
                  </Button>
                  <Button
                    className="flex-1"
                    onClick={() => {
                      setUploadedImage(transformedImage)
                      setTransformedImage(null)
                    }}
                  >
                    Use as New Input
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
        <ActivityHistory />
      </div>
    ),
    [
      dragActive,
      uploadedImage,
      handleDrag,
      handleDrop,
      triggerFileInput,
      handleImageUpload,
      selectedTransform,
      imageUploadProgress,
      transformImage,
      loading,
      transformedImage,
    ],
  )

  const TextToSpeechPage = useMemo(
    () => (
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[calc(100vh-8rem)]">
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Text to Speech</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Textarea
                name="text"
                placeholder="Enter text to convert to speech..."
                value={ttsForm.values.text}
                onChange={ttsForm.handleChange}
                rows={4}
                maxLength={1000}
              />
              <div className="text-right text-sm text-gray-500">{ttsForm.values.text.length}/1000 characters</div>

              <Select value={selectedVoice} onValueChange={setSelectedVoice}>
                <SelectTrigger>
                  <SelectValue placeholder="Select voice" />
                </SelectTrigger>
                <SelectContent>
                  {availableVoices.length > 0 ? (
                    availableVoices.map((voice) => (
                      <SelectItem key={voice.name} value={voice.name}>
                        {voice.display_name}
                      </SelectItem>
                    ))
                  ) : (
                    <SelectItem value="loading" disabled>
                      Loading voices...
                    </SelectItem>
                  )}
                </SelectContent>
              </Select>

              <Input
                name="music"
                placeholder="Search background music (optional)..."
                value={ttsForm.values.music}
                onChange={ttsForm.handleChange}
              />

              <Button
                onClick={generateSpeech}
                disabled={!ttsForm.values.text.trim() || !selectedVoice || loading}
                className="w-full"
              >
                {loading ? "Generating Speech..." : "Generate Speech"}
              </Button>
            </CardContent>
          </Card>

          {audioUrl && (
            <Card>
              <CardHeader>
                <CardTitle>Audio Player</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Main mixed audio */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium">Mixed Audio (Speech with Music)</h3>
                  <audio
                    ref={audioRef}
                    src={audioUrl}
                    controls
                    className="w-full"
                    onPlay={() => setIsPlaying(true)}
                    onPause={() => setIsPlaying(false)}
                    onEnded={() => setIsPlaying(false)}
                  />
                  <Button variant="outline" size="sm" onClick={downloadAudio} disabled={!audioUrl}>
                    <Download size={16} className="mr-2" />
                    Download Mixed Audio
                  </Button>
                </div>

                {/* Background music only */}
                {ttsResponse?.background_music?.audio_url && (
                  <div className="space-y-2">
                    <h3 className="text-sm font-medium">Background Music Only</h3>
                    <audio
                      src={`${API_BASE}${ttsResponse.background_music.audio_url}`}
                      controls
                      className="w-full"
                    />
                    <div className="text-sm text-gray-500">
                      Track: {ttsResponse.background_music.name}
                    </div>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={() => {
                        const a = document.createElement("a");
                        a.href = `${API_BASE}${ttsResponse.background_music.audio_url}`;
                        a.download = `background_${Date.now()}.wav`;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                      }}
                    >
                      <Download size={16} className="mr-2" />
                      Download Background Music
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </div>
        <ActivityHistory />
      </div>
    ),
    [
      ttsForm.values.text,
      ttsForm.values.music,
      ttsForm.handleChange,
      selectedVoice,
      availableVoices,
      generateSpeech,
      loading,
      audioUrl,
      downloadAudio,
      audioBlob,
    ],
  )

  const FileUploadPage = useMemo(
    () => (
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[calc(100vh-8rem)]">
        <div className="lg:col-span-2">
          <Card className="h-full">
            <CardHeader>
              <CardTitle>File Upload</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center bg-gray-50">
                <Upload className="mx-auto h-12 w-12 text-gray-400" />
                <p className="mt-2 text-sm text-gray-600">Drag and drop files here or click to upload</p>
                <p className="text-xs text-gray-500 mt-1">Supports: TXT, PDF, DOC, DOCX, JPG, PNG, MP3, WAV</p>
                <input
                  type="file"
                  accept=".txt,.pdf,.doc,.docx,.jpg,.png,.mp3,.wav"
                  className="hidden"
                  id="file-upload"
                />
                <label htmlFor="file-upload" className="cursor-pointer">
                  <Button className="mt-4" variant="outline">
                    Choose Files
                  </Button>
                </label>
              </div>
            </CardContent>
          </Card>
        </div>
        <ActivityHistory />
      </div>
    ),
    [],
  )

  // Simplified render function
  const renderCurrentPage = () => {
    if (!user && currentPage !== "login" && currentPage !== "signup") {
      return LoginPage
    }

    switch (currentPage) {
      case "login":
        return LoginPage
      case "signup":
        return SignupPage
      case "photo-editor":
        return PhotoEditorPage
      case "text-to-speech":
        return TextToSpeechPage
      default:
        return LoginPage
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">{renderCurrentPage()}</main>
    </div>
  )
}