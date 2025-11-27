import { useState, useEffect, lazy } from "react";
import { Alert, Typography, Button, FormHelperText, Box } from "@mui/material";
import { PlaceholdersAndVanishInput } from "../Ascternity/Input/VanishInput";
import { HugeiconsIcon } from "@hugeicons/react";
import { Mail02Icon, ArrowLeftIcon } from "@hugeicons/core-free-icons";
import { apiClient } from "../../api/api";
import { mixpanelLoginFunction } from "../../mixpanel/funcs";

const OtpInputComponent = lazy(() => import("./OtpInput"));

const Login = ({ children }) => {
  const [showLogin, setShowLogin] = useState(true);
  const [errorMessage, setErrorMessage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [email, setEmail] = useState("");
  const [otp, setOtp] = useState("");
  const [emailError, setEmailError] = useState("");
  const [otpError, setOtpError] = useState("");
  const [showOtpScreen, setShowOtpScreen] = useState(false);

  // Validate email format
  const validateEmail = (value) => {
    const emailRegex = /^\S+@\S+\.\S+$/;
    if (!value) {
      return "Email is required";
    }
    if (!emailRegex.test(value)) {
      return "Please enter a valid email address";
    }
    return "";
  };

  // Validate OTP format
  const validateOtp = (value) => {
    if (!value) {
      return "OTP is required";
    }
    if (!/^\d{6}$/.test(value)) {
      return "OTP must be a 6-digit number";
    }
    return "";
  };

  // Handle email input change
  const handleEmailChange = (e) => {
    const value = e.target.value;
    setEmail(value);
    setEmailError(validateEmail(value));
  };

  // Handle OTP input change
  const handleOtpChange = (value) => {
    setOtp(value);
    setOtpError(validateOtp(value));
    if (value?.length === 6) {
      handleOtpSubmit(value);
    } else {
      setOtpError("");
    }
  };

  // Handle email submit
  const handleEmailSubmit = async () => {
    const emailValidationError = validateEmail(email);
    if (emailValidationError) {
      setEmailError(emailValidationError);
      return;
    }
    setIsLoading(true);
    try {
      await apiClient.post("/auth/login", { email });
      setShowOtpScreen(true);
    } catch (error) {
      setErrorMessage(
        error.response?.data?.message || "Failed to send OTP. Please try again."
      );
    } finally {
      setIsLoading(false);
    }
  };

  // Handle OTP submit
  const handleOtpSubmit = async (otp) => {
    const otpValidationError = validateOtp(otp);
    if (otpValidationError) {
      setOtpError(otpValidationError);
      return;
    }

    setIsLoading(true);
    try {
      const response = await apiClient.post("/auth/verify", { email, otp });
      const { access_token, user_id, email: userEmail } = response.data;

      // Store auth data in localStorage
      localStorage.setItem("accessToken", access_token);
      localStorage.setItem("userId", user_id);
      localStorage.setItem("email", userEmail);
      setShowLogin(false);
      setShowOtpScreen(false);
      mixpanelLoginFunction({ email: userEmail });
    } catch (error) {
      setOtpError(
        error.response?.data?.message || "Invalid OTP. Please try again."
      );
    } finally {
      setIsLoading(false);
    }
  };

  // Handle back to email screen
  const handleBackToEmail = () => {
    setShowOtpScreen(false);
    setOtp("");
    setOtpError("");
    setErrorMessage(null);
  };

  // Check if already logged in
  useEffect(() => {
    const hasEmail = localStorage.getItem("email");
    const hasAccessToken = localStorage.getItem("accessToken");
    const hasUserId = localStorage.getItem("userId");

    if (hasEmail && hasAccessToken && hasUserId) {
      setShowLogin(false);
    }
  }, []);

  return (
    <>
      {showLogin && false ? (
        <div className="flex gap-4 h-full p-4 bg-gradient-to-br from-white to-blue-50">
          {/* Left Section - Login Form */}
          <div className="w-full lg:w-1/2 flex flex-col justify-center items-center p-8 rounded-[32px]">
            <div className="w-full flex flex-col items-center gap-4 max-w-md">
              <div className="flex flex-col gap-4 items-center">
                <div
                  className="flex justify-center items-center p-3 rounded-full bg-white border border-[#FFFFFF80] w-fit"
                  style={{
                    boxShadow: "0px 7.2px 36px 0px rgba(87, 162, 237, 0.50)",
                  }}
                >
       
                </div>
                <p className="text-xl md:text-2xl text-[#192944] font-semibold w-full text-center">
                                    <span className="font-normal text-[#57A2ED]">
                    Jarvis.
                  </span>
                </p>
              </div>
              <div className="flex flex-col gap-4 mt-10 mb-4 items-center">
                <p className="text-[#192944] text-xl md:text-2xl font-medium">
                  Welcome Back!
                </p>
                <p className="text-[#525068] text-sm md:text-base text-center">
                  Please enter your email address so we can send you a one-time
                  password (OTP) to log in.
                </p>
              </div>
              {showOtpScreen ? (
                <>
                  <div className="w-full flex flex-col items-center gap-4">
                    <div className="text-center mb-2">
                      <Typography variant="subtitle1" color="primary">
                        Enter your 6-digit OTP
                      </Typography>
                      <Typography variant="caption" color="textSecondary">
                        A verification code has been sent to your email
                      </Typography>
                    </div>

                    <OtpInputComponent
                      value={otp}
                      onChange={handleOtpChange}
                      disabled={isLoading}
                      maxLength={6}
                    />

                    <Button
                      variant="contained"
                      color="primary"
                      onClick={() => handleOtpSubmit(otp)}
                      disabled={otp.length !== 6 || isLoading}
                      sx={{
                        width: "100%",
                        borderRadius: "25px",
                        padding: "10px",
                        marginTop: "12px",
                        textTransform: "none",
                        fontWeight: "bold",
                      }}
                    >
                      {isLoading ? "Verifying..." : "Verify OTP"}
                    </Button>
                  </div>

                  {otpError && (
                    <FormHelperText error> {otpError}</FormHelperText>
                  )}
                  <div className="flex items-center justify-between w-full">
                    <Button
                      onClick={handleBackToEmail}
                      startIcon={
                        <HugeiconsIcon icon={ArrowLeftIcon} color="#7E7E7E" />
                      }
                      sx={{ textTransform: "none", color: "#7E7E7E" }}
                    >
                      Back
                    </Button>
                  </div>
                </>
              ) : (
                <>
                  <PlaceholdersAndVanishInput
                    placeholders={[
                      "hello@hatypo.studio",
                      "Enter your email to sign up",
                      "Unlock Jarvis",
                      "Start your AI journey",
                    ]}
                    onChange={handleEmailChange}
                    onSubmit={(e) => {
                      e.preventDefault();
                      handleEmailSubmit();
                    }}
                    startIcon={Mail02Icon}
                    loading={isLoading}
                    value={email}
                    aria-label="Email address"
                    aria-required="true"
                  />
                  {emailError && (
                    <FormHelperText error>{emailError}</FormHelperText>
                  )}
                </>
              )}
              {errorMessage && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  {errorMessage}
                </Alert>
              )}
            </div>
          </div>

          {/* Right Section - Promotional Content */}
          <div className="w-1/2 hidden lg:block relative rounded-[32px]">
            <img
              alt="login illustration"
              src="https://i.ibb.co/sptNYqTD/Illustration.png"
              className="w-full h-full object-cover rounded-[32px]"
            />
          </div>
        </div>
      ) : (
        children
      )}
    </>
  );
};

export default Login;
