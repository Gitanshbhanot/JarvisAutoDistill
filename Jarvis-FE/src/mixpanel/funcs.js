import mixpanel from "mixpanel-browser";
import { mixpanelToken } from "..";

// Initialize Mixpanel
export const mixpanelInit = () => {
  if (mixpanelToken) {
    mixpanel.init(mixpanelToken, {
      api_host: "https://drdun9bya6vw5.cloudfront.net",
      persistence: "localStorage",
      track_pageview: true,
    });
  }
};

// Identify user via email and register super property
export const mixpanelIdentify = async ({ email = "" }) => {
  if (mixpanelToken && email) {
    mixpanel.identify(email);
    mixpanel.register({
      email: email,
    });
  }
};

// Track user login and set people properties
export const mixpanelLoginFunction = async ({ email = "" }) => {
  if (mixpanelToken && email) {
    mixpanel.identify(email);
    mixpanel.people.set({
      email: email,
    });
    mixpanel.register({
      email: email,
    });
    mixpanel.track("User login", {
      email: email,
    });
  }
};

// Track logout
export const mixpanelTrackLogOut = () => {
  if (mixpanelToken) {
    mixpanel.track("Log out");
    mixpanel.reset();
  }
};