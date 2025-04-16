"use client";

import { useState } from "react"
import Link from "next/link"
import { Upload, Camera, ImageIcon, X, AlertCircle, ChevronRight } from "lucide-react"

export default function DetectPage() {
  const [activeTab, setActiveTab] = useState("upload")
  const [previewImage, setPreviewImage] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && (file.type === "image/jpeg" || file.type === "image/png")) {
      setSelectedFile(file); // Save the actual file for API
      const reader = new FileReader();
      reader.onload = () => {
        setPreviewImage(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };
  
  /*const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file && (file.type === "image/jpeg" || file.type === "image/png")) {
      const reader = new FileReader()
      reader.onload = () => {
        setPreviewImage(reader.result)
      }
      reader.readAsDataURL(file)
    }
  }*/

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file && (file.type === "image/jpeg" || file.type === "image/png")) {
      const reader = new FileReader()
      reader.onload = () => {
        setPreviewImage(reader.result)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleCameraCapture = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      setTimeout(() => {
        setPreviewImage("/api/placeholder/640/480")
        stream.getTracks().forEach(track => track.stop())
      }, 500)
    } catch (err) {
      console.error("Error accessing camera:", err)
      alert("Unable to access camera. Please ensure you have given camera permissions.")
    }
  }

  const handleRemoveImage = () => {
    setPreviewImage(null)
  }
  const handleAnalyze = async () => {
    if (!selectedFile) return;
  
    setIsAnalyzing(true);
    const formData = new FormData();
    formData.append("file", selectedFile);
  
    try {
      const response = await fetch("http://127.0.0.1:8000/predict/", {
        method: "POST",
        body: formData,
      });
  
      const data = await response.json();
      console.log("Prediction result:", data);
  
      // Redirect or store prediction for use in /results
      localStorage.setItem("prediction", JSON.stringify(data));
      window.location.href = "/results";
    } catch (error) {
      console.error("Prediction failed:", error);
      alert("Something went wrong during prediction.");
    } finally {
      setIsAnalyzing(false);
    }
  };
  

  /*const handleAnalyze = () => {

    setIsAnalyzing(true)
    // Simulate analysis process
    setTimeout(() => {
      setIsAnalyzing(false)
      // In a real app, you would redirect to results page or show results here
      window.location.href = "/results"
    }, 3000)
  }*/

  return (
    <div className="min-h-screen bg-black text-white flex flex-col">
      <nav className="w-full p-4 flex justify-between items-center">
        <Link href="/" className="text-lg font-medium hover:text-gray-300 transition-colors">
          Lung/Colon Cancer Detection model
        </Link>
        <div className="flex items-center gap-6">
          <Link href="/features" className="hover:text-gray-300 transition-colors">
            Features
          </Link>
          <Link href="/contact" className="hover:text-gray-300 transition-colors">
            Contact
          </Link>
          <Link href="/past-scans" className="hover:text-gray-300 transition-colors">
            Past Scans
          </Link>
        </div>
      </nav>

      <main className="flex-1 flex flex-col items-center px-4 py-12">
        <div className="w-full max-w-3xl">
          <h1 className="text-4xl md:text-5xl font-light mb-4 text-center">Detect Cancer Markers</h1>
          <p className="text-xl text-gray-400 mb-8 text-center">
            Upload a medical scan image or take a photo to analyze for potential cancer markers
          </p>

          <div className="bg-gray-900 rounded-lg overflow-hidden mb-8">
            <div className="flex border-b border-gray-800">
              <button
                className={`flex-1 py-4 text-center ${
                  activeTab === "upload" ? "bg-black text-white" : "text-gray-400 hover:text-white"
                }`}
                onClick={() => setActiveTab("upload")}
              >
                <Upload className="inline-block w-5 h-5 mr-2" />
                Upload Image
              </button>
              <button
                className={`flex-1 py-4 text-center ${
                  activeTab === "camera" ? "bg-black text-white" : "text-gray-400 hover:text-white"
                }`}
                onClick={() => setActiveTab("camera")}
              >
                <Camera className="inline-block w-5 h-5 mr-2" />
                Take Photo
              </button>
            </div>

            <div className="p-6">
              {previewImage ? (
                <div className="relative">
                  <img src={previewImage} alt="Preview" className="w-full h-64 object-contain rounded-lg mb-4" />
                  <button
                    onClick={handleRemoveImage}
                    className="absolute top-2 right-2 bg-black bg-opacity-50 p-1 rounded-full hover:bg-opacity-70 transition-opacity"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>
              ) : activeTab === "upload" ? (
                <div
                  className={`border-2 border-dashed rounded-lg p-8 text-center ${
                    isDragging ? "border-white bg-gray-800" : "border-gray-700"
                  }`}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                >
                  <ImageIcon className="w-12 h-12 mx-auto mb-4 text-gray-500" />
                  <p className="mb-4">Drag and drop your image here, or click to browse</p>
                  <input
                    type="file"
                    id="file-upload"
                    className="hidden"
                    accept="image/png, image/jpeg"
                    onChange={handleFileChange}
                  />
                  <label
                    htmlFor="file-upload"
                    className="bg-white text-black hover:bg-gray-200 px-6 py-2 rounded-full text-base transition-colors focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-black inline-block cursor-pointer"
                  >
                    Browse Files
                  </label>
                  <p className="mt-4 text-sm text-gray-400">Supported formats: JPEG, PNG</p>
                </div>
              ) : (
                <div className="text-center">
                  <Camera className="w-12 h-12 mx-auto mb-4 text-gray-500" />
                  <p className="mb-4">Allow access to your camera to take a photo</p>
                  <button
                    onClick={handleCameraCapture}
                    className="bg-white text-black hover:bg-gray-200 px-6 py-2 rounded-full text-base transition-colors focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-black inline-block"
                  >
                    Access Camera
                  </button>
                  <p className="mt-4 text-sm text-gray-400">
                    For best results, ensure good lighting and a clear view of the scan
                  </p>
                </div>
              )}

              {previewImage && (
                <button
                  onClick={handleAnalyze}
                  disabled={isAnalyzing}
                  className="w-full bg-white text-black hover:bg-gray-200 px-6 py-3 rounded-full text-base transition-colors focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-black flex items-center justify-center"
                >
                  {isAnalyzing ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-black mr-3"></div>
                      Analyzing...
                    </>
                  ) : (
                    <>
                      Analyze Image
                      <ChevronRight className="ml-2 w-4 h-4" />
                    </>
                  )}
                </button>
              )}
            </div>
          </div>

          <div className="border border-gray-800 rounded-lg p-6">
            <div className="flex items-start">
              <AlertCircle className="w-5 h-5 text-gray-400 mr-3 mt-1" />
              <div>
                <h3 className="font-medium mb-2">Important Information</h3>
                <p className="text-gray-400 text-sm mb-2">
                  This AI detection model is designed to assist in the identification of potential cancer markers but is
                  not a replacement for professional medical diagnosis.
                </p>
                <p className="text-gray-400 text-sm">
                  Always consult with a qualified healthcare provider regarding your scan results and for proper medical
                  advice.
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>

      <footer className="p-4 text-center text-sm text-gray-400">
        *Results may be inaccurate. Please contact a medical professional for a complete diagnosis.
      </footer>
    </div>
  )
}