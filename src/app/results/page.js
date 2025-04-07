"use client";

import Link from "next/link"
import React, { useState, useEffect } from 'react';
export default function ResultsPage() {
  // Mock results - in a real app, these would come from API/props
  
  const [result, setResult] = useState(null);

  useEffect(() => {
    const storedResult = localStorage.getItem("prediction");
    if (storedResult) {
      setResult(JSON.parse(storedResult));
    }
  }, []);

  if (!result) return <p>Loading...</p>;

  return (
    <div className="text-white p-6">
      <h1 className="text-3xl mb-4">Prediction Results</h1>
      <p className="text-xl">Class: {result.prediction}</p>
      <p className="text-xl">Confidence: {(result.confidence * 100).toFixed(2)}%</p>
    </div>
  );



  /* ts DOES NOT work pls fix
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

      <main className="flex-1 flex flex-col items-center justify-center px-4">
        <h1 className="text-4xl md:text-5xl font-light mb-8">Scan Results</h1>
        <p className="text-gray-400 mb-12">Scan completed on {results.scanDate}</p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 md:gap-16 w-full max-w-4xl">
          <div className="flex flex-col items-center">
            <div className="relative w-48 h-48 mb-6">
              <div className="absolute inset-0 rounded-full border-4 border-gray-700"></div>
              <div
                className="absolute inset-0 rounded-full border-4 border-white"
                style={{
                  clipPath: `polygon(0 0, 100% 0, 100% 100%, 0% 100%)`,
                  opacity: results.lungCancerChance / 100,
                }}
              ></div>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-5xl font-bold">{results.lungCancerChance}%</span>
              </div>
            </div>
            <h2 className="text-2xl font-medium">Chance of Lung Cancer</h2>
            <p className="text-gray-400 mt-2 text-center">
              {results.lungCancerChance < 15
                ? "Low risk detected. Regular check-ups recommended."
                : "Elevated risk detected. Consult a medical professional."}
            </p>
          </div>

          <div className="flex flex-col items-center">
            <div className="relative w-48 h-48 mb-6">
              <div className="absolute inset-0 rounded-full border-4 border-gray-700"></div>
              <div
                className="absolute inset-0 rounded-full border-4 border-white"
                style={{
                  clipPath: `polygon(0 0, 100% 0, 100% 100%, 0% 100%)`,
                  opacity: results.colonCancerChance / 100,
                }}
              ></div>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-5xl font-bold">{results.colonCancerChance}%</span>
              </div>
            </div>
            <h2 className="text-2xl font-medium">Chance of Colon Cancer</h2>
            <p className="text-gray-400 mt-2 text-center">
              {results.colonCancerChance < 10
                ? "Low risk detected. Regular check-ups recommended."
                : "Elevated risk detected. Consult a medical professional."}
            </p>
          </div>
        </div>

        <div className="flex flex-col sm:flex-row gap-4 mt-16">
          <Link
            href="/detect"
            className="bg-white text-black hover:bg-gray-200 px-8 py-3 rounded-full text-base transition-colors focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-black"
          >
            New Scan
          </Link>
          <Link
            href="/past-scans"
            className="border border-white text-white hover:bg-white hover:text-black px-8 py-3 rounded-full text-base transition-colors focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-black"
          >
            View Past Scans
          </Link>
        </div>
      </main>

      <footer className="p-4 text-center text-sm text-gray-400 mt-8">
        *Results may be inaccurate. Please contact a medical professional for a complete diagnosis.
      </footer>
    </div>
  )*/
}

