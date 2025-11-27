import { Avatar, Dialog, IconButton } from "@mui/material";
import { mixpanelTrackLogOut } from "../mixpanel/funcs";
import { useWindowSize } from "@uidotdev/usehooks";
import { useState } from "react";

const handleLogout = () => {
  mixpanelTrackLogOut();
  localStorage.removeItem("email");
  localStorage.removeItem("userId");
  window.location.reload();
};

const avatars = ["man1", "man2", "female1", "female2"];
const randomAvatar = avatars[Math.floor(Math.random() * avatars.length)];
const avatarSrc = `https://img.favpng.com/11/21/25/iron-man-cartoon-avatar-superhero-icon-png-favpng-jrRBMJQjeUwuteGtBce87yMxz.jpg`;

const DialogContent = ({ open, setOpen, email }) => {
  return (
    <Dialog
      open={open}
      onClose={() => {
        setOpen(false);
      }}
      sx={{
        borderRadius: "32px",
        "& .MuiDialog-paper": {
          borderRadius: "32px",
        },
      }}
    >
      <div className="flex flex-col gap-4 px-2 py-4 items-center w-[50dvw] rounded-[32px]">
        <Avatar
          sx={{
            width: "72px",
            height: "72px",
          }}
          alt="Profile Image"
          src={avatarSrc}
        />
        <p className="text-xl md:text-2xl font-semibold">
          Hello{" "}
          <span className="text-[#57A2ED] font-semibold capitalize">
            {email?.split("@")?.[0]}!
          </span>
        </p>
        <p
          className="text-[#687485] text-sm hover:underline cursor-pointer px-2 py-1 rounded border bg-blue-50"
          onClick={handleLogout}
        >
          Logout
        </p>
      </div>
    </Dialog>
  );
};

const Profile = ({ isMobile = false }) => {
  const [open, setOpen] = useState(false);
  const { width } = useWindowSize();
  const email = localStorage.getItem("email");

  return (
    <div className="flex gap-2 items-center relative">
      {width < 768 ? (
        <IconButton
          onClick={() => {
            setOpen(true);
          }}
          size="small"
        >
          <Avatar
            alt="Profile Image"
            src={avatarSrc}
            sx={{
              height: "32px",
              width: "32px",
            }}
          />
        </IconButton>
      ) : (
        <div className="flex items-center justify-between p-1 rounded-full border border-[#DDEEFF]">
          <Avatar
            sx={{
              width: "42px",
              height: "42px",
            }}
            alt="Profile Image"
            src={avatarSrc}
          />
        </div>
      )}
      {width >= 768 && (
        <div className="flex flex-col gap-0">
          <p className="text-sm font-semibold">
            Welcome{" "}
            <span className="text-[#57A2ED] text-base font-semibold capitalize">
              {/* {email?.split("@")?.[0]}! */}Back!
            </span>
          </p>
          {/* <p
            className="text-[#687485] text-sm hover:underline cursor-pointer"
            onClick={handleLogout}
          >
            Logout
          </p> */}
        </div>
      )}
      <DialogContent
        open={open}
        setOpen={setOpen}
        email={email}
        avatarSrc={avatarSrc}
      />
    </div>
  );
};

const NavBar = () => {
  return (
    <div className="flex gap justify-between gap-2 items-center pt-2 px-2 pb-2 md:pt-4 md:px-6 border-b border-blue-10 z-20">
      <div className="flex gap-4 items-center">
        <div className="flex gap-2 items-center">
          <div className="rounded-full relative">
            <img
              alt="jarvis"
              src="https://wallpapers.com/images/high/iron-man-arc-reactor-graphic-tabwo742fk9net6d.png"
              className="h-7 w-7 md:h-10 md:w-10 animate-[spin_3s_linear_infinite]"
            />
            <div className="absolute flex inset-0 justify-center items-center">
              <div className="bg-blue-300 animate-ping w-2 h-2 md:w-5 md:h-5 rounded-full" />
            </div>
          </div>

          <p className="text-xl md:text-2xl font-semibold">
            <span className="text-[#57A2ED] font-semmibold">Jarvis.</span>
          </p>
        </div>
      </div>
      <div className="flex gap-2 items-center">
        <Profile />
      </div>
    </div>
  );
};

export default NavBar;
