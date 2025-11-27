import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import App from "./App";
import reportWebVitals from "./reportWebVitals";
import { BrowserRouter } from "react-router-dom";
import { createTheme, ThemeProvider } from "@mui/material";
import { mixpanelInit } from "./mixpanel/funcs";
import { ToastProvider } from "./components/Toast/Toast";

// Keep only environment variables needed for VideoAnalysis
export const baseURL = import.meta.env.VITE_BASE_URL;
export const mixpanelToken = import.meta.env.VITE_MIXPANEL_TOKEN || null;

if (import.meta.env.PROD) {
  console.log = () => {};
  console.error = () => {};
  console.warn = () => {};
  console.debug = () => {};
  window.onerror = function () {
    return true; // suppresses the default browser error
  };
  // To catch unhandled promise rejections
  window.onunhandledrejection = function () {
    return true;
  };
  mixpanelInit();
}

const root = ReactDOM.createRoot(document.getElementById("root"));

const MuiTheme = createTheme({
  typography: {
    fontFamily: '"Plus Jakarta Sans", sans-serif',
    fontSize: 13,
  },
  components: {
    MuiDataGrid: {
      defaultProps: {
        slotProps: {
          panel: {
            sx: {
              "& .MuiDataGrid-panelWrapper": {
                maxWidth: "calc(100vw - 3rem)",
              },
            },
          },
        },
      },
    },
  },
});

root.render(
  <BrowserRouter>
    <ThemeProvider theme={MuiTheme}>
      <ToastProvider>
        <App />
      </ToastProvider>
    </ThemeProvider>
  </BrowserRouter>
);

reportWebVitals();
