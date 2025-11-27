import axios from "axios";

const BASE_URL = import.meta.env.VITE_BASE_URL;
const JARVIS_API_BASE =
  import.meta.env.VITE_JARVIS_API || "http://0.0.0.0:8000";

const getAuthToken = () => localStorage.getItem("accessToken");

const authInterceptor = (config) => {
  if (config.url !== "/auth/login" && config.url !== "/auth/verify") {
    const token = getAuthToken();
    if (token) {
      config.headers = config.headers || {};
      config.headers.Authorization = `Bearer ${token}`;
    }
  }
  return config;
};

const errorInterceptor = (error) => {
  if (error.response?.status === 401) {
    localStorage.clear();
    window.location.href = "/";
  }
  return Promise.reject(error);
};

// JSON client
export const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: { "Content-Type": "application/json", "X-Tenant-Id": "jindal" },
  timeout: 5 * 60 * 1000,
});

// FormData client (let Axios auto-handle Content-Type)
export const apiClientFormData = axios.create({
  baseURL: BASE_URL,
  headers: { "X-Tenant-Id": "jindal" },
  timeout: 5 * 60 * 1000,
});

// Attach interceptors
apiClient.interceptors.request.use(authInterceptor, Promise.reject);
apiClient.interceptors.response.use((res) => res, errorInterceptor);

apiClientFormData.interceptors.request.use(authInterceptor, Promise.reject);
apiClientFormData.interceptors.response.use((res) => res, errorInterceptor);

// Jarvis API clients (no auth by default)
export const jarvisApiClient = axios.create({
  baseURL: JARVIS_API_BASE,
  paramsSerializer: { indexes: null },
  headers: { "Content-Type": "application/json" },
  timeout: 5 * 60 * 1000,
});

export const jarvisApiClientFormData = axios.create({
  baseURL: JARVIS_API_BASE,
  // Let Axios set multipart boundary automatically
  timeout: 5 * 60 * 1000,
});

jarvisApiClient.interceptors.response.use((res) => res, errorInterceptor);
jarvisApiClientFormData.interceptors.response.use((res) => res, errorInterceptor);
