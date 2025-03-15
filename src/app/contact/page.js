import Link from "next/link"
import { Mail, MapPin, Phone, Send, ChevronRight } from "lucide-react"

export default function ContactPage() {
  const contactMethods = [
    {
      icon: <Mail className="w-6 h-6" />,
      title: "Email",
      description: "Our team typically responds within 24 hours",
      details: "support@cancer-detection-ai.com",
    },
    {
      icon: <Phone className="w-6 h-6" />,
      title: "Phone",
      description: "Mon-Fri from 8am to 5pm EST",
      details: "+1 (555) 123-4567",
    },
    {
      icon: <MapPin className="w-6 h-6" />,
      title: "Office",
      description: "Come visit our headquarters",
      details: "101 Innovation Drive, Cambridge, MA 02142",
    },
  ]

  return (
    <div className="min-h-screen bg-black text-white flex flex-col">
      <nav className="w-full p-4 flex justify-between items-center">
        <Link href="/" className="text-lg font-medium hover:text-gray-300 transition-colors">
          Lung/Colon Cancer Detection model
        </Link>
        <div className="flex items-center gap-6">
          <Link href="/features" className="hover:text-gray-300 transition-colors font-medium">
            Features
          </Link>
          <Link href="/contact" className="hover:text-gray-300 transition-colors" aria-current="page">
            Contact
          </Link>
          <Link href="/login" className="hover:text-gray-300 transition-colors">
            Login/Sign Up
          </Link>
        </div>
      </nav>

      <main className="flex-1 flex flex-col items-center px-4 py-12">
        <div className="w-full max-w-6xl">
          <h1 className="text-4xl md:text-5xl font-light mb-4 text-center">Contact Us</h1>
          <p className="text-xl text-gray-400 mb-16 text-center max-w-3xl mx-auto">
            Have questions about our AI cancer detection technology or need assistance? Our team is here to help.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
            {contactMethods.map((method, index) => (
              <div key={index} className="border border-gray-800 rounded-lg p-6 hover:border-white transition-colors">
                <div className="mb-4 text-white">{method.icon}</div>
                <h2 className="text-xl font-medium mb-2">{method.title}</h2>
                <p className="text-gray-400 mb-4">{method.description}</p>
                <p className="text-white">{method.details}</p>
              </div>
            ))}
          </div>

          <div className="border border-gray-800 rounded-lg p-8">
            <h2 className="text-2xl font-light mb-6">Send us a message</h2>
            <form className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label htmlFor="name" className="block text-sm font-medium text-gray-400 mb-2">
                    Your Name
                  </label>
                  <input
                    type="text"
                    id="name"
                    className="w-full bg-gray-900 border border-gray-800 rounded-lg p-3 text-white focus:outline-none focus:ring-2 focus:ring-white focus:border-transparent"
                    placeholder="John Doe"
                  />
                </div>
                <div>
                  <label htmlFor="email" className="block text-sm font-medium text-gray-400 mb-2">
                    Email Address
                  </label>
                  <input
                    type="email"
                    id="email"
                    className="w-full bg-gray-900 border border-gray-800 rounded-lg p-3 text-white focus:outline-none focus:ring-2 focus:ring-white focus:border-transparent"
                    placeholder="john@example.com"
                  />
                </div>
              </div>
              <div>
                <label htmlFor="subject" className="block text-sm font-medium text-gray-400 mb-2">
                  Subject
                </label>
                <input
                  type="text"
                  id="subject"
                  className="w-full bg-gray-900 border border-gray-800 rounded-lg p-3 text-white focus:outline-none focus:ring-2 focus:ring-white focus:border-transparent"
                  placeholder="How can we help you?"
                />
              </div>
              <div>
                <label htmlFor="message" className="block text-sm font-medium text-gray-400 mb-2">
                  Message
                </label>
                <textarea
                  id="message"
                  rows={4}
                  className="w-full bg-gray-900 border border-gray-800 rounded-lg p-3 text-white focus:outline-none focus:ring-2 focus:ring-white focus:border-transparent"
                  placeholder="Tell us about your question or concern..."
                />
              </div>
              <div>
                <button
                  type="submit"
                  className="bg-white text-black hover:bg-gray-200 px-8 py-3 rounded-full text-base transition-colors focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-black inline-flex items-center"
                >
                  Send Message
                  <Send className="ml-2 w-4 h-4" />
                </button>
              </div>
            </form>
          </div>

          <div className="mt-16 text-center">
            <h2 className="text-2xl font-light mb-6">Ready to experience our technology?</h2>
            <Link
              href="/detect"
              className="bg-white text-black hover:bg-gray-200 px-8 py-3 rounded-full text-base transition-colors focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-black inline-flex items-center"
            >
              Start a Scan
              <ChevronRight className="ml-2 w-4 h-4" />
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