"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

const ORGAN_MAP = {
  lung_scc: "Lung Squamous Cell Carcinoma",
  lung_aca:  "Lung Adenocarcinoma",
  lung_n: "Lung Benign Tissue",
  colon: "Colon Cancer",
  colon_n: "Colon Benign Tissue"
};

function parseResult(prediction, confidence) {
  const isLungscc  = prediction.startsWith("lung_scc");
  const isLungaca = prediction.startsWith("lung_aca")
  const isColon = prediction.startsWith("colon_aca");
  
  const isColonBenign = prediction.startsWith("colon_n");
  const isLungBenign = prediction.startsWith("lung_n")

  if (isLungscc) return {organKey: "lung_scc", pct: Math.min(99.99,confidence) }
  if (isLungaca)  return { organKey: "lung_aca",  pct: Math.min(99.99,confidence) };
  if (isColon) return { organKey: "colon", pct: Math.min(99.99,confidence) };
  if (isColonBenign) return { organKey: "colon_n", pct: 0 }; 
  if (isLungBenign) return { organKey: "lung_n", pct: 0 };
  return { organKey: null, pct: 0 };
}

export default function ResultsPage() {
  const [result, setResult] = useState(null);

  useEffect(() => {
    try {
      const stored = localStorage.getItem("prediction");
      if (stored) setResult(JSON.parse(stored));
    } catch (err) {
      console.error("Could not parse prediction", err);
    }
  }, []);

  if (!result) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <p className="text-xl">No scan result found. Please run a new scan.</p>
      </div>
    );
  }

  const { organKey, pct } = parseResult(result.prediction, result.confidence);
  const organLabel = ORGAN_MAP[organKey] ?? "Unknown";
  const advice =
    pct < 15
      ? "Low risk detected. Regular check‑ups recommended."
      : "Elevated risk detected. Consult a medical professional.";
  const scanDate = new Date().toLocaleDateString();

  return (
    <div className="min-h-screen bg-black text-white flex flex-col">
      {/* ---------- NAV ---------- */}
      <nav className="w-full p-4 flex justify-between items-center">
        <Link href="/" className="text-lg font-medium hover:text-gray-300">
          Cancer Detection Model
        </Link>
        <div className="flex items-center gap-6">
          <Link href="/features" className="hover:text-gray-300">
            Features
          </Link>
          <Link href="/contact" className="hover:text-gray-300">
            Contact
          </Link>
          <Link href="/login"  className="hover:text-gray-300">
            Login
          </Link>
          <Link href="/signup" className="hover:text-gray-300">
            Sign&nbsp;Up
          </Link>
        </div>
      </nav>

      {/* ---------- MAIN ---------- */}
      <main className="flex-1 flex flex-col items-center justify-center px-4">
        <h1 className="text-4xl md:text-5xl font-light mb-8">Scan Results</h1>
        <p className="text-gray-400 mb-12">Scan completed on {scanDate}</p>

        {/* single circle card */}
        <div className="flex flex-col items-center">
          <div className="relative w-48 h-48 mb-6">
            <div className="absolute inset-0 rounded-full border-4 border-gray-700"></div>
            {/* white sector showing % */}
            <div
              className="absolute inset-0 rounded-full border-4 border-white"
              style={{
                clipPath: "polygon(0 0, 100% 0, 100% 100%, 0% 100%)",
                opacity: pct / 100,
              }}
            ></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-5xl font-bold">
                {pct.toFixed(2)}%
              </span>
            </div>
          </div>
          <h2 className="text-2xl font-medium">{organLabel}</h2>
          <p className="text-gray-400 mt-2 text-center max-w-xs">{advice}</p>
        </div>

        {/* button */}
        <div className="mt-16">
          <Link
            href="/detect"
            className="bg-white text-black hover:bg-gray-200 px-8 py-3 rounded-full text-base transition-colors focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-black"
          >
            New Scan
          </Link>
        </div>
      </main>

      {/* ---------- FOOTER ---------- */}
      <footer className="p-4 text-center text-sm text-gray-400 mt-8">
        *Results may be inaccurate. Please contact a medical professional for a complete diagnosis.
      </footer>
    </div>
  );
}

/*export default function ResultsPage() {
  
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
      <p className="text-xl">Confidence: {(result.confidence).toFixed(2)}%</p>
    </div>
  );


}*/

