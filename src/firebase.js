// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getAuth } from "firebase/auth"; 
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyBs2YW0Bh0ntfzHztjYjyITEt3pa3sZY1Y",
  authDomain: "lung-and-colon-cancer-project.firebaseapp.com",
  projectId: "lung-and-colon-cancer-project",
  storageBucket: "lung-and-colon-cancer-project.firebasestorage.app",
  messagingSenderId: "1002508301768",
  appId: "1:1002508301768:web:b5a0704b9ba5ba5b2dfcfd",
  measurementId: "G-V7XVG81RDC"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
  
export { app };
  