/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#e8f5ee',
          100: '#c8e6d4',
          200: '#a8d4b8',
          300: '#4a9d6e',
          400: '#2d5a3f',
          500: '#1a3d2e',
        },
        lavender: {
          50: '#f3f0fa',
          100: '#d4cce8',
          200: '#a594d4',
          300: '#7c6aad',
          400: '#4a3d6e',
        },
        surface: '#f4f7fa',
        card: '#ffffff',
        border: '#e8ecf0',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      borderRadius: {
        card: '16px',
        pill: '24px',
        input: '12px',
      },
      animation: {
        'float': 'float 3s ease-in-out infinite',
        'pulse-slow': 'pulse 3s ease-in-out infinite',
        'fade-in': 'fadeIn 0.5s ease-out',
        'slide-up': 'slideUp 0.4s ease-out',
        'breathe': 'breathe 19s ease-in-out infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-8px)' },
        },
        fadeIn: {
          from: { opacity: '0', transform: 'translateY(8px)' },
          to: { opacity: '1', transform: 'none' },
        },
        slideUp: {
          from: { opacity: '0', transform: 'translateY(16px)' },
          to: { opacity: '1', transform: 'none' },
        },
        breathe: {
          '0%, 100%': { transform: 'scale(0.8)', opacity: '0.4' },
          '21%': { transform: 'scale(1.4)', opacity: '0.9' },
          '58%': { transform: 'scale(1.4)', opacity: '0.9' },
        },
      },
    },
  },
  plugins: [],
}
