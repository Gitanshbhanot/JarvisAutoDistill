import React from "react";
import {
  Grid,
  Card,
  CardContent,
  CardActionArea,
  Skeleton,
  Alert,
  Stack,
  Pagination,
  Box,
  Typography,
} from "@mui/material";
import { HugeiconsIcon } from "@hugeicons/react";
import { Alert01Icon } from "@hugeicons/core-free-icons";

const ImageGrid = ({
  items,
  filteredCount,
  page,
  pageCount,
  onPageChange,
  isLoading,
  viewSource,
  query,
  onOpenPreview,
}) => {
  if (isLoading) {
    return (
      <Grid container spacing={2}>
        {Array.from({ length: 24 }).map((_, i) => (
          <Grid key={i} item xs={12} sm={6} md={4} lg={3}>
            <Card>
              <CardContent>
                <Skeleton variant="text" width="70%" />
                <Skeleton variant="rectangular" sx={{ mt: 1 }} height={140} />
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    );
  }

  if (filteredCount === 0) {
    return (
      <Alert
        icon={<HugeiconsIcon icon={Alert01Icon} />}
        severity="info"
        sx={{ my: 4 }}
      >
        {query ? (
          <>No results for “{query}”. Try a different search.</>
        ) : (
          <>No images found in this {viewSource} view.</>
        )}
      </Alert>
    );
  }

  return (
    <>
      <Grid container spacing={2}>
        {items.map((path) => (
          <Grid key={path} item xs={12} sm={6} md={4} lg={3}>
            <Card elevation={1}>
              <CardActionArea onClick={() => onOpenPreview(path)}>
                <CardContent
                  sx={{
                    minHeight: 120,
                    display: "flex",
                    flexDirection: "column",
                    gap: 1,
                  }}
                >
                  <Typography
                    variant="body2"
                    color="text.secondary"
                    title={path}
                    sx={{
                      display: "-webkit-box",
                      WebkitLineClamp: 2,
                      WebkitBoxOrient: "vertical",
                      overflow: "hidden",
                    }}
                  >
                    {path}
                  </Typography>
                  <Box
                    sx={{
                      mt: 1,
                      border: "1px solid",
                      borderColor: "divider",
                      borderRadius: 1,
                      height: 120,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      bgcolor: "background.default",
                    }}
                  >
                    <Typography variant="caption" color="text.secondary">
                      Click to preview
                    </Typography>
                  </Box>
                </CardContent>
              </CardActionArea>
            </Card>
          </Grid>
        ))}
      </Grid>

      {pageCount > 1 && (
        <Stack alignItems="center" mt={3}>
          <Pagination
            page={page}
            count={pageCount}
            onChange={(_, p) => onPageChange(p)}
            color="primary"
            shape="rounded"
          />
        </Stack>
      )}
    </>
  );
};

export default ImageGrid;
