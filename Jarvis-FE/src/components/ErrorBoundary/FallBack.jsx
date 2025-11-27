import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Typography,
  Box,
  Button,
} from "@mui/material";
import { HugeiconsIcon } from "@hugeicons/react";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";

export const Fallback = ({ error = "", errorInfo, showErrors = true }) => {
  return (
    <Box
      sx={{
        width: "100%",
        height: "100%",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        backgroundColor: "grey.50",
        fontFamily: "Plus Jakarta Sans",
      }}
    >
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          gap: 2,
          alignItems: "center",
          justifyContent: "center",
          textAlign: "center",
          p: 4,
          borderRadius: 2,
          boxShadow: 3,
          backgroundColor: "white",
          width: { xs: "90%", md: 600 },
        }}
      >
        <Typography
          variant="h4"
          component="p"
          color="textPrimary"
          fontWeight="bold"
        >
          Oops! Something went wrong.
        </Typography>
        <Typography
          variant="body1"
          component="div"
          color="textSecondary"
          fontWeight="medium"
        >
          We track these errors automatically, but if the problem persists, feel
          free to contact us. In the meantime, try refreshing the page.
        </Typography>

        {showErrors && (
          <Accordion sx={{ width: "100%" }}>
            <AccordionSummary
              expandIcon={<HugeiconsIcon icon={ArrowDown01Icon} />}
              sx={{ backgroundColor: "grey.200", borderRadius: 1, px: 2 }}
            >
              <Typography
                sx={{ flex: 1, textAlign: "left", fontWeight: "medium" }}
              >
                Error details
              </Typography>
            </AccordionSummary>
            <AccordionDetails
              sx={{
                display: "flex",
                flexDirection: "column",
                gap: 2,
                border: 1,
                borderColor: "grey.300",
                borderRadius: 1,
                p: 2,
                fontSize: "0.875rem",
                backgroundColor: "grey.100",
                maxHeight: 200,
                overflowY: "auto",
              }}
            >
              <Typography fontWeight="bold">Error:</Typography>
              <Typography color="error">{error}</Typography>
              <Typography fontWeight="bold">Component Stack Trace:</Typography>
              <Typography color="textSecondary">
                {errorInfo?.componentStack}
              </Typography>
            </AccordionDetails>
          </Accordion>
        )}
        <Button
          variant="contained"
          color="primary"
          onClick={() => window.location.reload()}
        >
          Refresh page
        </Button>
      </Box>
    </Box>
  );
};
