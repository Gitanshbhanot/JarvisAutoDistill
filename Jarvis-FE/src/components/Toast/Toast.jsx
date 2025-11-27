import React, { useState, createContext, useContext } from "react";
import { Snackbar, Alert, AlertTitle } from "@mui/material";

const ToastContext = createContext();

export const ToastProvider = ({ children }) => {
  const [toast, setToast] = useState({
    open: false,
    title: "",
    description: "",
    status: "info",
    duration: 4000,
    isClosable: true,
    position: { vertical: "top", horizontal: "right" },
  });

  const showToast = ({
    title = "",
    description = "",
    status = "info",
    duration = 4000,
    isClosable = true,
    position = "bottom-right",
  }) => {
    // Convert Chakra position (like "top", "bottom-right") to MUI anchorOrigin format
    const anchorOrigin = getAnchorOrigin(position);

    setToast({
      open: true,
      title,
      description,
      status,
      duration,
      isClosable,
      position: anchorOrigin,
    });
  };

  const closeToast = () => {
    setToast((prev) => ({
      ...prev,
      open: false,
    }));
  };

  // Helper function to map Chakra position props to MUI's anchorOrigin
  const getAnchorOrigin = (position) => {
    switch (position) {
      case "top":
        return { vertical: "top", horizontal: "center" };
      case "top-left":
        return { vertical: "top", horizontal: "left" };
      case "top-right":
        return { vertical: "top", horizontal: "right" };
      case "bottom":
        return { vertical: "bottom", horizontal: "center" };
      case "bottom-left":
        return { vertical: "bottom", horizontal: "left" };
      case "bottom-right":
      default:
        return { vertical: "bottom", horizontal: "right" };
    }
  };

  return (
    <ToastContext.Provider value={showToast}>
      {children}
      <Snackbar
        open={toast.open}
        // autoHideDuration={toast.duration}
        autoHideDuration={6000}
        onClose={closeToast}
        // anchorOrigin={toast.position}
        anchorOrigin={{ vertical: "top", horizontal: "right" }}
      >
        <Alert
          onClose={closeToast}
          severity={toast.status}
          variant="filled"
        >
          {toast.title && <AlertTitle>{toast.title}</AlertTitle>}
          {toast.description}
        </Alert>
      </Snackbar>
    </ToastContext.Provider>
  );
};

export const useToast = () => useContext(ToastContext);
