import React from "react";
import { Fallback } from "./FallBack";

export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: "", errorInfo: "" };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({
      error: `${error?.name} - ${error?.message}`,
      errorInfo: errorInfo,
    });
  }

  render() {
    const { error, errorInfo } = this.state;
    if (error) {
      return <Fallback error={error} errorInfo={errorInfo} />;
    } else {
      return <>{this.props.children}</>;
    }
  }
}
