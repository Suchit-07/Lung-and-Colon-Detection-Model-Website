"use client";

import { useState } from "react"
import Link from "next/link"
import { Calendar, Filter, ChevronDown, Download, Eye } from "lucide-react"

export default function PastScansPage() {
  const [sortColumn, setSortColumn] = useState("date")
  const [sortDirection, setSortDirection] = useState("desc")
  
  // Sample scan history data
  const scans = [
    { id: 1, date: "2025-03-01", colonCancer: 0.8, lungCancer: 1.2, imagePath: "/scans/scan-001.jpg" },
    { id: 2, date: "2025-02-15", colonCancer: 1.4, lungCancer: 0.5, imagePath: "/scans/scan-002.jpg" },
    { id: 3, date: "2025-01-23", colonCancer: 0.3, lungCancer: 0.7, imagePath: "/scans/scan-003.jpg" },
    { id: 4, date: "2024-12-10", colonCancer: 2.1, lungCancer: 3.4, imagePath: "/scans/scan-004.jpg" },
    { id: 5, date: "2024-11-05", colonCancer: 0.5, lungCancer: 0.9, imagePath: "/scans/scan-005.jpg" },
  ]
  
  // Sort the scans based on the selected column and direction
  const sortedScans = [...scans].sort((a, b) => {
    if (sortColumn === "date") {
      return sortDirection === "asc" 
        ? new Date(a.date) - new Date(b.date)
        : new Date(b.date) - new Date(a.date)
    } else if (sortColumn === "colonCancer") {
      return sortDirection === "asc" ? a.colonCancer - b.colonCancer : b.colonCancer - a.colonCancer
    } else if (sortColumn === "lungCancer") {
      return sortDirection === "asc" ? a.lungCancer - b.lungCancer : b.lungCancer - a.lungCancer
    }
    return 0
  })
  
  // Handle sorting when a column header is clicked
  const handleSort = (column) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc")
    } else {
      setSortColumn(column)
      setSortDirection("asc")
    }
  }
  
  // Format date to be more readable
  const formatDate = (dateString) => {
    const date = new Date(dateString)
    return date.toLocaleDateString("en-US", { year: 'numeric', month: 'short', day: 'numeric' })
  }

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
          <Link href="/login" className="hover:text-gray-300 transition-colors">
            Logout
          </Link>
        </div>
      </nav>

      <main className="flex-1 flex flex-col items-center px-4 py-12">
        <div className="w-full max-w-6xl">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8">
            <div>
              <h1 className="text-3xl font-light mb-2">Your Scan History</h1>
              <p className="text-gray-400">Review your past scans and detection results</p>
            </div>
            <div className="mt-4 md:mt-0 flex gap-3">
              <Link
                href="/detect"
                className="bg-white text-black hover:bg-gray-200 px-6 py-2 rounded-full text-sm transition-colors focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-black"
              >
                New Scan
              </Link>
              <button className="border border-gray-800 hover:border-white px-3 py-2 rounded-full text-sm transition-colors focus:outline-none flex items-center">
                <Filter className="w-4 h-4 mr-1" />
                Filter
              </button>
            </div>
          </div>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-800">
                    <th 
                      className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider cursor-pointer"
                      onClick={() => handleSort("date")}
                    >
                      <div className="flex items-center">
                        <Calendar className="w-4 h-4 mr-1" />
                        Date
                        {sortColumn === "date" && (
                          <ChevronDown className={`w-4 h-4 ml-1 transition-transform ${sortDirection === "asc" ? "rotate-180" : ""}`} />
                        )}
                      </div>
                    </th>
                    <th 
                      className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider cursor-pointer"
                      onClick={() => handleSort("colonCancer")}
                    >
                      <div className="flex items-center">
                        Colon Cancer Risk
                        {sortColumn === "colonCancer" && (
                          <ChevronDown className={`w-4 h-4 ml-1 transition-transform ${sortDirection === "asc" ? "rotate-180" : ""}`} />
                        )}
                      </div>
                    </th>
                    <th 
                      className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider cursor-pointer"
                      onClick={() => handleSort("lungCancer")}
                    >
                      <div className="flex items-center">
                        Lung Cancer Risk
                        {sortColumn === "lungCancer" && (
                          <ChevronDown className={`w-4 h-4 ml-1 transition-transform ${sortDirection === "asc" ? "rotate-180" : ""}`} />
                        )}
                      </div>
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {sortedScans.map((scan, index) => (
                    <tr key={scan.id} className={`border-b border-gray-800 ${index % 2 === 0 ? "bg-gray-900" : "bg-gray-950"}`}>
                      <td className="px-6 py-4 whitespace-nowrap">
                        {formatDate(scan.date)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <div className="w-16 bg-gray-800 rounded-full h-2 mr-2">
                            <div 
                              className="bg-red-500 h-2 rounded-full" 
                              style={{ width: `${Math.min(scan.colonCancer * 10, 100)}%` }}
                            ></div>
                          </div>
                          <span className={scan.colonCancer > 2 ? "text-red-500" : "text-gray-300"}>
                            {scan.colonCancer.toFixed(1)}%
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <div className="w-16 bg-gray-800 rounded-full h-2 mr-2">
                            <div 
                              className="bg-red-500 h-2 rounded-full" 
                              style={{ width: `${Math.min(scan.lungCancer * 10, 100)}%` }}
                            ></div>
                          </div>
                          <span className={scan.lungCancer > 2 ? "text-red-500" : "text-gray-300"}>
                            {scan.lungCancer.toFixed(1)}%
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                        <div className="flex justify-end gap-2">
                          <button className="p-1 hover:bg-gray-800 rounded-full" title="View details">
                            <Eye className="w-4 h-4" />
                          </button>
                          <button className="p-1 hover:bg-gray-800 rounded-full" title="Download report">
                            <Download className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            {sortedScans.length === 0 ? (
              <div className="p-8 text-center">
                <p className="text-gray-400">No scan history found</p>
                <Link 
                  href="/detect"
                  className="mt-4 inline-block bg-white text-black hover:bg-gray-200 px-6 py-2 rounded-full text-sm transition-colors"
                >
                  Upload Your First Scan
                </Link>
              </div>
            ) : (
              <div className="p-4 text-sm text-gray-400 flex justify-between items-center">
                <p>Showing {sortedScans.length} scans</p>
                <div className="flex items-center gap-2">
                  <button className="px-3 py-1 border border-gray-800 rounded hover:border-white transition-colors">
                    Previous
                  </button>
                  <span className="px-3 py-1 bg-gray-800 rounded">1</span>
                  <button className="px-3 py-1 border border-gray-800 rounded hover:border-white transition-colors">
                    Next
                  </button>
                </div>
              </div>
            )}
          </div>
          
          <div className="mt-6 p-4 border border-gray-800 rounded-lg text-sm text-gray-400">
            <p>
              <strong className="text-white">Note:</strong> Risk percentages below 1% are generally considered low risk. 
              Any percentage above 2% may warrant a discussion with your healthcare provider. These AI predictions are for 
              informational purposes only and should not replace professional medical advice.
            </p>
          </div>
        </div>
      </main>

      <footer className="p-4 text-center text-sm text-gray-400">
        *Results may be inaccurate. Please contact a medical professional for a complete diagnosis.
      </footer>
    </div>
  )
}