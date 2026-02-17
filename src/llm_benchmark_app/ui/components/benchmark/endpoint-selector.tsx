import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Label } from "@/components/ui/label";
import { useEndpoints, type EndpointOut } from "@/lib/benchmark-api";
import { Button } from "@/components/ui/button";
import { Search, Server } from "lucide-react";

interface EndpointSelectorProps {
  selected: string[];
  onSelectionChange: (selected: string[]) => void;
}

export function EndpointSelector({
  selected,
  onSelectionChange,
}: EndpointSelectorProps) {
  const [search, setSearch] = useState("");
  const { data: endpoints, isLoading, error } = useEndpoints();

  const filtered = (endpoints ?? []).filter(
    (ep) =>
      ep.name.toLowerCase().includes(search.toLowerCase()) ||
      (ep.model_name ?? "").toLowerCase().includes(search.toLowerCase()),
  );

  const toggleEndpoint = (name: string) => {
    if (selected.includes(name)) {
      onSelectionChange(selected.filter((n) => n !== name));
    } else if (selected.length < 4) {
      onSelectionChange([...selected, name]);
    }
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Server className="h-4 w-4" />
          Select Endpoints
          <div className="ml-auto flex items-center gap-2">
            {selected.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                className="h-6 px-2 text-xs text-muted-foreground"
                onClick={() => onSelectionChange([])}
              >
                Deselect all
              </Button>
            )}
            <Badge variant="secondary">
              {selected.length}/4
            </Badge>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="relative">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search endpoints..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9 h-9"
          />
        </div>

        <div className="max-h-[280px] overflow-y-auto space-y-1 pr-1">
          {isLoading && (
            <div className="space-y-2">
              {[1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          )}

          {error && (
            <p className="text-sm text-destructive py-2">
              Failed to load endpoints. Check your workspace connection.
            </p>
          )}

          {!isLoading && filtered.length === 0 && (
            <p className="text-sm text-muted-foreground py-4 text-center">
              {search ? "No endpoints match your search" : "No serving endpoints found"}
            </p>
          )}

          {filtered.map((ep) => (
            <EndpointRow
              key={ep.name}
              endpoint={ep}
              checked={selected.includes(ep.name)}
              disabled={!selected.includes(ep.name) && selected.length >= 4}
              onToggle={() => toggleEndpoint(ep.name)}
            />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function EndpointRow({
  endpoint,
  checked,
  disabled,
  onToggle,
}: {
  endpoint: EndpointOut;
  checked: boolean;
  disabled: boolean;
  onToggle: () => void;
}) {
  const stateColor =
    endpoint.state === "READY"
      ? "bg-green-500"
      : endpoint.state === "NOT_READY"
        ? "bg-yellow-500"
        : "bg-gray-400";

  return (
    <Label
      className={`flex items-center gap-3 rounded-md border p-3 cursor-pointer transition-colors hover:bg-accent/50 ${
        checked ? "border-primary bg-accent/30" : ""
      } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
    >
      <Checkbox
        checked={checked}
        onCheckedChange={onToggle}
        disabled={disabled}
      />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium truncate">{endpoint.name}</span>
          <span
            className={`h-2 w-2 rounded-full shrink-0 ${stateColor}`}
            title={endpoint.state}
          />
        </div>
        {(endpoint.model_name || endpoint.task) && (
          <div className="flex items-center gap-2 mt-0.5">
            {endpoint.model_name && (
              <span className="text-xs text-muted-foreground truncate">
                {endpoint.model_name}
              </span>
            )}
            {endpoint.task && (
              <Badge variant="outline" className="text-[10px] h-4 px-1">
                {endpoint.task}
              </Badge>
            )}
          </div>
        )}
      </div>
    </Label>
  );
}
