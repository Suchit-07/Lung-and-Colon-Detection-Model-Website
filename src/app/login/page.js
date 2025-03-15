import Link from "next/link"
import { Lock, User, ArrowRight } from "lucide-react"

export default function LoginPage() {
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
            <h1 className="text-4xl font-light mb-4">Welcome Back</h1>
            <p className="text-gray-400">
              Log in to access your scan history and continue monitoring your health
            </p>
          </div>

          <div className="border border-gray-800 rounded-lg p-8">
            <div className="flex space-x-4 mb-6">
              <button className="flex-1 bg-white text-black py-3 rounded-md font-medium">
                Log In
              </button>
              <Link href="/signup" className="flex-1 border border-gray-800 text-center py-3 rounded-md hover:border-white transition-colors">
                Sign Up
              </Link>
            </div>

            <form className="space-y-6">
              <div>
                <label htmlFor="email" className="block text-sm font-medium text-gray-400 mb-2">
                  Email Address
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <User className="h-5 w-5 text-gray-500" />
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
                <div className="flex items-center justify-between mb-2">
                  <label htmlFor="password" className="block text-sm font-medium text-gray-400">
                    Password
                  </label>
                  
                </div>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Lock className="h-5 w-5 text-gray-500" />
                  </div>
                  <input
                    type="password"
                    id="password"
                    className="w-full bg-gray-900 border border-gray-800 rounded-lg pl-10 p-3 text-white focus:outline-none focus:ring-2 focus:ring-white focus:border-transparent"
                    placeholder="••••••••"
                  />
                </div>
              </div>

              <div className="flex items-center">
                <input
                  id="remember-me"
                  name="remember-me"
                  type="checkbox"
                  className="h-4 w-4 text-white bg-gray-900 border-gray-800 rounded focus:ring-white"
                />
                <label htmlFor="remember-me" className="ml-2 block text-sm text-gray-400">
                  Remember me for 30 days
                </label>
              </div>

              <button
                type="submit"
                className="w-full bg-white text-black hover:bg-gray-200 py-3 rounded-lg text-base transition-colors focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-black flex items-center justify-center"
              >
                Log In
                <ArrowRight className="ml-2 w-4 h-4" />
              </button>
            </form>

            <div className="mt-6 text-center">
              <span className="text-gray-400">Don't have an account?</span>{" "}
              <Link href="/signup" className="text-white hover:underline">
                Sign up
              </Link>
            </div>
          </div>

          <div className="mt-8 text-center text-sm text-gray-400">
            By logging in, you agree to our{" "}
            <Link href="/terms" className="text-white hover:underline">
              Terms of Service
            </Link>{" "}
            and{" "}
            <Link href="/privacy" className="text-white hover:underline">
              Privacy Policy
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