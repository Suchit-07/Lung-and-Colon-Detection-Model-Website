import Link from "next/link"
import { Brain, LineChart, Clock, Shield, Database, Award } from "lucide-react"

export default function FeaturesPage() {
  const features = [
    {
      icon: <Brain className="w-10 h-10" />,
      title: "Advanced AI Detection",
      description:
        "Our model uses state-of-the-art deep learning algorithms trained on thousands of medical images to detect potential cancer markers.",
    },
    {
      icon: <LineChart className="w-10 h-10" />,
      title: "High Accuracy",
      description:
        "Achieves over 90% accuracy in early detection, comparable to expert radiologists in controlled studies.",
    },
    {
      icon: <Clock className="w-10 h-10" />,
      title: "Rapid Results",
      description:
        "Get your scan results in seconds, not days or weeks, allowing for faster medical consultation if needed.",
    },
    {
      icon: <Shield className="w-10 h-10" />,
      title: "Privacy Focused",
      description:
        "Your medical data and scan images are encrypted and never shared with third parties without your explicit consent.",
    },
    {
      icon: <Database className="w-10 h-10" />,
      title: "Scan History",
      description:
        "Track your scan history over time to monitor changes and share results with your healthcare provider.",
    },
    {
      icon: <Award className="w-10 h-10" />,
      title: "Clinically Validated",
      description: "Our detection model has been validated through clinical trials and peer-reviewed research.",
    },
  ]

  return (
    <div className="min-h-screen bg-black text-white flex flex-col">
      <nav className="w-full p-4 flex justify-between items-center">
        <Link href="/" className="text-lg font-medium hover:text-gray-300 transition-colors">
          Lung/Colon Cancer Detection model
        </Link>
        <div className="flex items-center gap-6">
          <Link href="/features" className="hover:text-gray-300 transition-colors font-medium" aria-current="page">
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

      <main className="flex-1 flex flex-col items-center px-4 py-12">
        <div className="w-full max-w-6xl">
          <h1 className="text-4xl md:text-5xl font-light mb-4 text-center">Features</h1>
          <p className="text-xl text-gray-400 mb-16 text-center max-w-3xl mx-auto">
            Our AI-powered cancer detection model offers cutting-edge technology to help identify potential health
            concerns early.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <div key={index} className="border border-gray-800 rounded-lg p-6 hover:border-white transition-colors">
                <div className="mb-4 text-white">{feature.icon}</div>
                <h2 className="text-xl font-medium mb-2">{feature.title}</h2>
                <p className="text-gray-400">{feature.description}</p>
              </div>
            ))}
          </div>

          <div className="mt-16 text-center">
            <h2 className="text-2xl font-light mb-6">Ready to try it yourself?</h2>
            <Link
              href="/detect"
              className="bg-white text-black hover:bg-gray-200 px-8 py-3 rounded-full text-base transition-colors focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-black inline-block"
            >
              Start a Scan
            </Link>
          </div>
        </div>
      </main>

      <footer className="p-4 text-center text-sm text-gray-400">
        *Results may be inaccurate. Please contact a medical professional for a complete diagnosis.
      </footer>
    </div>
  )
}

