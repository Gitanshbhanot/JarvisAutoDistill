import React from "react";
import {
  Stack,
  Typography,
  Button,
  ToggleButton,
  ToggleButtonGroup,
  TextField,
  InputAdornment,
  Tooltip,
  IconButton,
} from "@mui/material";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  ArrowLeft01Icon,
  Search01Icon,
  RefreshIcon,
  Delete02Icon,
  Download01Icon,
  Video01Icon,
} from "@hugeicons/core-free-icons";

const TopBar = ({
  datasetId,
  viewSource,
  onViewSourceChange,
  query,
  onQueryChange,
  isLoading,
  onRefresh,
  onBack,
  onDelete,
  onDownload,
  isDownloading,
  onVideoDownload,
  isVideoDownloading,
  onReannotate,
}) => {
  return (
    <Stack
      direction="row"
      alignItems="center"
      justifyContent="space-between"
      mb={2}
      gap={2}
    >
      <Stack direction="row" alignItems="center" gap={1.5} flexWrap="wrap">
        <Button
          variant="outlined"
          startIcon={<HugeiconsIcon icon={ArrowLeft01Icon} />}
          onClick={onBack}
        >
          Back
        </Button>
        <Typography
          variant="h5"
          fontWeight={700}
          sx={{ wordBreak: "break-word" }}
        >
          Dataset: {datasetId}
        </Typography>

        <ToggleButtonGroup
          size="small"
          value={viewSource}
          exclusive
          onChange={(_, val) => val && onViewSourceChange(val)}
          aria-label="View source"
        >
          <ToggleButton value="annotated" aria-label="Annotated">
            Annotated
          </ToggleButton>
          <ToggleButton value="raw" aria-label="Raw">
            Raw
          </ToggleButton>
        </ToggleButtonGroup>
      </Stack>

      <Stack direction="row" gap={1} alignItems="center" flexWrap="wrap">
        <TextField
          size="small"
          placeholder="Search images"
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <HugeiconsIcon icon={Search01Icon} />
              </InputAdornment>
            ),
          }}
        />
        <Tooltip title="Refresh">
          <span>
            <IconButton onClick={onRefresh} disabled={isLoading}>
              <HugeiconsIcon icon={RefreshIcon} />
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title="Download dataset">
          <span>
            <IconButton
              onClick={onDownload}
              disabled={isLoading || isDownloading}
            >
              <HugeiconsIcon icon={Download01Icon} />
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title="Download annotated video">
          <span>
            <IconButton
              onClick={onVideoDownload}
              disabled={isLoading || isVideoDownloading}
            >
              <HugeiconsIcon icon={Video01Icon} />
            </IconButton>
          </span>
        </Tooltip>
        {onReannotate && (
          <Button
            variant="outlined"
            onClick={onReannotate}
            disabled={isLoading}
          >
            Re-annotate
          </Button>
        )}
        <Tooltip title="Delete dataset">
          <span>
            <Button
              color="error"
              variant="outlined"
              startIcon={<HugeiconsIcon icon={Delete02Icon} />}
              onClick={onDelete}
              disabled={isLoading}
            >
              Delete
            </Button>
          </span>
        </Tooltip>
      </Stack>
    </Stack>
  );
};

export default TopBar;