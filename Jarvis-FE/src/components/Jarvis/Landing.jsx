import { AiNetworkIcon, DatabaseAddIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useLocation, useNavigate } from "react-router-dom";

const Landing = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const tasks = [
    {
      name: "Add data",
      description: "Add data to Jarvis",
      icon: <HugeiconsIcon icon={DatabaseAddIcon} />,
      onClick: () =>
        navigate("/data/home", {
          state: { prevPath: location.pathname + location.search },
        }),
    },
    {
      name: "Train model",
      description: "Train model via Jarvis",
      icon: <HugeiconsIcon icon={AiNetworkIcon} />,
      onClick: () =>
        navigate("/model/home", {
          state: { prevPath: location.pathname + location.search },
        }),
    },
  ];
  return (
    <div className="h-full w-full flex flex-col gap-4 justify-center items-center">
      <div className="flex flex-col gap-2 items-center">
        <p className="text-4xl font-bold">Welcome to Jarvis</p>
        <p className="text-xl text-gray-500">
          Jarvis is a smart assistant that can help you with your tasks.
        </p>
      </div>
      <div className="grid grid-cols-2 gap-4">
        {tasks.map((task) => (
          <div
            className="flex flex-col gap-2 items-center justify-center hover:shadow-md bg-white p-2 rounded-lg border cursor-pointer w-[200px] h-[200px]"
            key={task?.name}
            onClick={task.onClick}
          >
            <div className="flex items-center gap-1">
              {task.icon}
              <p className="text-xl font-bold">{task.name}</p>
            </div>

            <p className="text-sm text-gray-500">{task.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Landing;
