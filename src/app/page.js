import Link from "next/link"

export default function Home() {
  return (
    <div className="min-h-[80vh] bg-black text-white flex flex-col">
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
            Login/Sign Up
          </Link>
        </div>
      </nav>

      <main className="flex-1 flex flex-col items-center justify-center text-center px-4 max-w-4xl mx-auto">
        <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight mb-6">
          An AI model for early lung and colon cancer detection
        </h1>
        <p className="text-lg md:text-xl text-gray-300 mb-8">Identify symptoms of lung and colon cancer early</p>
        <Link
          href="/detect"
          className="bg-white text-black hover:bg-gray-200 font-medium px-8 py-3 rounded-full text-base transition-colors focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-black"
        >
          Detect Today
        </Link>
      </main>

      <footer className="p-4 text-center text-sm text-gray-400">
        *Results may be inaccurate. Please contact a medical professional for a complete diagnosis.
      </footer>
    </div>
  )
}

