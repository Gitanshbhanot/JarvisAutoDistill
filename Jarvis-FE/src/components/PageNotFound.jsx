import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { HugeiconsIcon } from "@hugeicons/react";
import { Home13Icon, Search01Icon } from "@hugeicons/core-free-icons";
import { Button } from "@mui/material";

const PageNotFound = () => {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <div className="w-full h-[90vh] flex flex-col rounded items-center justify-center text-center">
      <img
        src="/websiteUnderConstruction.svg"
        alt="Page Not Found"
        className="w-1/2 max-w-md mb-8"
      />
      <h1 className="text-2xl font-semibold text-gray-700 mb-4">
        Oops! Page Not Found
      </h1>
      <p className="text-gray-500 mb-6">
        The page you are looking for might be under construction or does not
        exist.
      </p>
      <Button
        onClick={() =>
          navigate("/home", {
            state: {
              prevPath: location?.pathname + location?.search,
            },
          })
        }
        sx={{
          minWidth: "140px",
        }}
        variant="outlined"
        startIcon={<HugeiconsIcon icon={Search01Icon} />}
      >
        Go to Home
      </Button>
    </div>
  );
};

export default PageNotFound;
