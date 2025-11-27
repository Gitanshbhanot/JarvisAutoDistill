/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Plus Jakarta Sans', 'Helvetica', 'sans-serif'],
        orbitron: ['Orbitron', 'Helvetica', 'sans-serif'],
      },
    },
  },
  plugins: [],
}