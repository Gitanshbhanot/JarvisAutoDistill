import { Navigate } from "react-router-dom";
import { lazy } from "react";
const PageNotFound = lazy(() => import("./components/PageNotFound"));
const Landing = lazy(() => import("./components/Jarvis/Landing"));
const DataHome = lazy(() => import("./components/Jarvis/Data/DataHome"));
const DataDetail = lazy(() => import("./components/Jarvis/Data/DataDetail"));
const ModelHome = lazy(() => import("./components/Jarvis/Model/ModelHome"));
const InferenceTest = lazy(() =>
  import("./components/Jarvis/Model/InferenceTest")
);

export const routes = [
  {
    path: "/",
    element: <Navigate to="/home" />,
    role: ["USER"],
  },
  {
    path: "/home",
    element: <Landing />,
    role: ["USER"],
  },
  // data pages
  {
    path: "/data/home",
    element: <DataHome />,
    role: ["USER"],
  },
  {
    path: "/data/:datasetId",
    element: <DataDetail />,
    role: ["USER"],
  },
  // model pages
  {
    path: "/model/home",
    element: <ModelHome />,
    role: ["USER"],
  },
  {
    path: "/model/inference/:modelTimestamp",
    element: <InferenceTest />,
    role: ["USER"],
  },
  {
    path: "*",
    element: <PageNotFound />,
    role: ["USER"],
  },
];
