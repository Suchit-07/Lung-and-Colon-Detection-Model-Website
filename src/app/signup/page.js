import Link from "next/link"
import { Lock, User, Mail, ArrowRight, CheckCircle } from "lucide-react"

export default function SignupPage() {
  const benefits = [
    "Store and access your scan history",
    "Receive personalized health insights",
    "Share results with your healthcare provider",
    "Get notified about important updates"
  ]

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
          <Link href="/login" className="hover:text-gray-300 transition-colors font-medium" aria-current="page">
            Login/Sign Up
          </Link>
        </div>
      </nav>

      <main className="flex-1 flex items-center justify-center px-4 py-12">
        <div className="w-full max-w-md">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-light mb-4">Create an Account</h1>
            <p className="text-gray-400">
              Join our platform to access advanced cancer detection technology
            </p>
          </div>

          <div className="border border-gray-800 rounded-lg p-8">
            <div className="flex space-x-4 mb-6">
              <Link href="/login" className="flex-1 border border-gray-800 text-center py-3 rounded-md hover:border-white transition-colors">
                Log In
              </Link>
              <button className="flex-1 bg-white text-black py-3 rounded-md font-medium">
                Sign Up
              </button>
            </div>

            <form className="space-y-6">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label htmlFor="first-name" className="block text-sm font-medium text-gray-400 mb-2">
                    First Name
                  </label>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <User className="h-5 w-5 text-gray-500" />
                    </div>
                    <input
                      type="text"
                      id="first-name"
                      className="w-full bg-gray-900 border border-gray-800 rounded-lg pl-10 p-3 text-white focus:outline-none focus:ring-2 focus:ring-white focus:border-transparent"
                      placeholder="John"
                    />
                  </div>
                </div>
                <div>
                  <label htmlFor="last-name" className="block text-sm font-medium text-gray-400 mb-2">
                    Last Name
                  </label>
                  <input
                    type="text"
                    id="last-name"
                    className="w-full bg-gray-900 border border-gray-800 rounded-lg p-3 text-white focus:outline-none focus:ring-2 focus:ring-white focus:border-transparent"
                    placeholder="Doe"
                  />
                </div>
              </div>

              <div>
                <label htmlFor="email" className="block text-sm font-medium text-gray-400 mb-2">
                  Email Address
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Mail className="h-5 w-5 text-gray-500" />
                  </div>
                  <input
                    type="email"
                    id="email"
                    className="w-full bg-gray-900 border border-gray-800 rounded-lg pl-10 p-3 text-white focus:outline-none focus:ring-2 focus:ring-white focus:border-transparent"
                    placeholder="your@email.com"
                  />
                </div>
              </div>
              
              <div>
                <label htmlFor="password" className="block text-sm font-medium text-gray-400 mb-2">
                  Password
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Lock className="h-5 w-5 text-gray-500" />
                  </div>
                  <input
                    type="password"
                    id="password"
                    className="w-full bg-gray-900 border border-gray-800 rounded-lg pl-10 p-3 text-white focus:outline-none focus:ring-2 focus:ring-white focus:border-transparent"
                    placeholder="Min. 8 characters"
                  />
                </div>
                <p className="mt-1 text-xs text-gray-400">Password must be at least 8 characters</p>
              </div>

              <div>
                <label htmlFor="confirm-password" className="block text-sm font-medium text-gray-400 mb-2">
                  Confirm Password
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Lock className="h-5 w-5 text-gray-500" />
                  </div>
                  <input
                    type="password"
                    id="confirm-password"
                    className="w-full bg-gray-900 border border-gray-800 rounded-lg pl-10 p-3 text-white focus:outline-none focus:ring-2 focus:ring-white focus:border-transparent"
                    placeholder="Confirm password"
                  />
                </div>
              </div>

              <div className="flex items-center">
                <input
                  id="terms"
                  name="terms"
                  type="checkbox"
                  className="h-4 w-4 text-white bg-gray-900 border-gray-800 rounded focus:ring-white"
                  required
                />
                <label htmlFor="terms" className="ml-2 block text-sm text-gray-400">
                  I agree to the{" "}
                  <Link href="/terms" className="text-white hover:underline">
                    Terms of Service
                  </Link>{" "}
                  and{" "}
                  <Link href="/privacy" className="text-white hover:underline">
                    Privacy Policy
                  </Link>
                </label>
              </div>

              <button
                type="submit"
                className="w-full bg-white text-black hover:bg-gray-200 py-3 rounded-lg text-base transition-colors focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-black flex items-center justify-center"
              >
                Create Account
                <ArrowRight className="ml-2 w-4 h-4" />
              </button>
            </form>
          </div>

          <div className="mt-8 border border-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-medium mb-4">Account Benefits</h3>
            <ul className="space-y-3">
              {benefits.map((benefit, index) => (
                <li key={index} className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-white mr-2 flex-shrink-0 mt-0.5" />
                  <span className="text-gray-400">{benefit}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </main>

      <footer className="p-4 text-center text-sm text-gray-400">
        *Results may be inaccurate. Please contact a medical professional for a complete diagnosis.
      </footer>
    </div>
  )
}