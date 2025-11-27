import "./App.css";
import React, {
  createContext,
  useContext,
  Suspense,
  useEffect,
  useState,
} from "react";
import { Route, Routes } from "react-router-dom";
import { ErrorBoundary } from "./components/ErrorBoundary/ErrorBoundary";
import { routes } from "./Route";
import { useWindowSize } from "@uidotdev/usehooks";
import { mixpanelIdentify } from "./mixpanel/funcs";
import Login from "./components/Login/Login";
import NavBar from "./components/NavBar";

const AppWideContext = createContext();

export const useAppWideContext = () => useContext(AppWideContext);

function App() {
  const [uploadProgress, setUploadProgress] = useState({
    uploading: false,
    chatId: null,
    chatName: null,
  });
  const email = localStorage.getItem("email") || "";

  return (
    <AppWideContext.Provider value={{ uploadProgress, setUploadProgress }}>
      <div className="relative h-[100dvh] min-w-screen w-full font-sans flex flex-col gap-0 bg-gradient-to-br from-white to-blue-100">
        <Login>
          <NavBar />
          <div className="px-2 pb-4 pt-2 md:p-4 md:px-6 flex-grow h-0 overflow-y-auto overflow-x-hidden z-10">
            <Routes>
              {routes.map((route, idx) => (
                <Route
                  key={idx}
                  path={route.path}
                  element={
                    <ErrorBoundary>
                      <Suspense
                        fallback={<h1 className="mt-10">Loading...</h1>}
                      >
                        {route.element}
                      </Suspense>
                    </ErrorBoundary>
                  }
                />
              ))}
            </Routes>
          </div>
        </Login>
        {/* {!email && (
          <img
            alt="ellipse"
            src="/planetEffect.svg"
            className="absolute inset-0 h-full w-full object-fill"
          />
        )} */}
      </div>
    </AppWideContext.Provider>
  );
}

export default App;
