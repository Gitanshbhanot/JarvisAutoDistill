import React, { useState } from "react";
import { useParams } from "react-router-dom";
import {
  Container,
  Stack,
  Typography,
  Button,
  TextField,
  Box,
  Alert,
  CircularProgress,
} from "@mui/material";
import { jarvisApiClientFormData } from "../../../api/api"; // Assume this is your API client
import { useToast } from "../../Toast/Toast";

const InferenceTest = () => {
  const { modelTimestamp } = useParams();
  const toast = useToast();

  const [image, setImage] = useState(null);
  const [confidence, setConfidence] = useState(0.25);
  const [isTesting, setIsTesting] = useState(false);
  const [annotatedImage, setAnnotatedImage] = useState(null);
  const [detectionInfo, setDetectionInfo] = useState(null);
  const [error, setError] = useState(null);

  const handleTest = async () => {
    if (!image) {
      toast({
        title: "Please upload an image",
        status: "error",
      });
      return;
    }

    setIsTesting(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("image", image);
      form.append("confidence", confidence);
      form.append("model_timestamp", modelTimestamp);

      const { data } = await jarvisApiClientFormData.post(
        "/api/inference",
        form
      );
      if (!data.ok) throw new Error(data.error || "Inference failed");

      setAnnotatedImage(`data:image/png;base64,${data.image.base64}`);
      setDetectionInfo(data.detection_info);
    } catch (e) {
      setError(e?.message || "Inference failed");
      toast({
        title: "Inference failed",
        status: "error",
        description: e?.message,
      });
    } finally {
      setIsTesting(false);
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h5" fontWeight={700} mb={3}>
        Test Inference for Model {modelTimestamp}
      </Typography>

      <Stack gap={2}>
        <Button variant="outlined" component="label">
          Upload Image
          <input
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => {
              console.log(e.target.files[0]);
              setImage(e.target.files[0]);
            }}
          />
        </Button>
        {image && <Typography>{image.name}</Typography>}

        <TextField
          label="Confidence Threshold"
          type="number"
          value={confidence}
          onChange={(e) => setConfidence(Number(e.target.value))}
          fullWidth
          inputProps={{ step: 0.05, min: 0, max: 1 }}
        />

        <Button
          variant="contained"
          onClick={handleTest}
          disabled={isTesting || !image}
        >
          Run Inference
        </Button>

        {isTesting && <CircularProgress />}

        {error && <Alert severity="error">{error}</Alert>}

        {annotatedImage && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="h6">Annotated Image</Typography>
            <img
              src={annotatedImage}
              alt="Annotated"
              style={{ maxWidth: "100%" }}
            />
          </Box>
        )}

        {detectionInfo && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="h6">Detection Info</Typography>
            <Typography whiteSpace="pre-wrap">{detectionInfo}</Typography>
          </Box>
        )}
      </Stack>
    </Container>
  );
};

export default InferenceTest;
