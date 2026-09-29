import {
  useState,
  type ComponentPropsWithoutRef,
  type KeyboardEvent,
  type MouseEvent,
  type ReactNode,
} from "react";

import { cn } from "./utils";

export type Tone = "neutral" | "sky" | "emerald" | "amber" | "rose" | "violet";

const toneClasses: Record<Tone, string> = {
  neutral: "border-zinc-200 bg-white text-zinc-700",
  sky: "border-sky-200 bg-sky-50 text-sky-800",
  emerald: "border-emerald-200 bg-emerald-50 text-emerald-800",
  amber: "border-amber-200 bg-amber-50 text-amber-800",
  rose: "border-rose-200 bg-rose-50 text-rose-800",
  violet: "border-violet-200 bg-violet-50 text-violet-800",
};

export function Panel({
  title,
  description,
  actions,
  children,
  className,
}: {
  title?: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={cn(
        "gap-0 rounded-lg border border-zinc-200 bg-white py-0 text-zinc-950 shadow-sm ring-0",
        className,
      )}
    >
      {(title || description || actions) && (
        <header className="flex flex-col gap-3 border-b border-zinc-200 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            {title && <h2 className="text-sm font-semibold text-zinc-950">{title}</h2>}
            {description && (
              <p className="mt-1 text-sm text-zinc-600">
                {description}
              </p>
            )}
          </div>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

export function Badge({
  children,
  tone = "neutral",
  className,
}: {
  children: ReactNode;
  tone?: Tone;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex min-h-6 rounded-md border px-2 py-0.5 text-xs font-medium shadow-none",
        toneClasses[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function StatCard({
  label,
  value,
  detail,
  tone = "neutral",
}: {
  label: ReactNode;
  value: ReactNode;
  detail?: ReactNode;
  tone?: Tone;
}) {
  return (
    <div className={cn("gap-1 rounded-lg border p-3 shadow-none", toneClasses[tone])}>
      <p className="text-xs font-medium uppercase tracking-normal opacity-75">
        {label}
      </p>
      <p className="mt-1 text-xl font-semibold text-zinc-950">{value}</p>
      {detail && (
        <p className="mt-1 text-xs leading-normal opacity-75">
          {detail}
        </p>
      )}
    </div>
  );
}

export function EmptyState({ children = "No results" }: { children?: ReactNode }) {
  return (
    <div className="min-h-24 rounded-lg border border-dashed border-zinc-300 bg-zinc-50 px-4 py-6 text-sm text-zinc-500">
      <p className="text-sm text-zinc-500">{children}</p>
    </div>
  );
}

export interface DataTableColumn<T> {
  key: string;
  header: ReactNode;
  cell: (row: T, index: number) => ReactNode;
  className?: string;
  headerClassName?: string;
}

export function DataTable<T>({
  rows,
  columns,
  getRowKey,
  empty = "No rows",
  onRowClick,
  rowClassName,
}: {
  rows: T[];
  columns: Array<DataTableColumn<T>>;
  getRowKey: (row: T, index: number) => string;
  empty?: ReactNode;
  onRowClick?: (row: T, index: number) => void;
  rowClassName?: (row: T, index: number) => string | false | null | undefined;
}) {
  if (rows.length === 0) {
    return <EmptyState>{empty}</EmptyState>;
  }

  const handleRowKeyDown = (event: KeyboardEvent<HTMLTableRowElement>, row: T, index: number) => {
    if (!onRowClick || (event.key !== "Enter" && event.key !== " ")) {
      return;
    }
    event.preventDefault();
    onRowClick(row, index);
  };

  return (
    <table className="min-w-full text-left text-sm">
      <thead className="border-b border-zinc-200 text-xs uppercase text-zinc-500">
        <tr className="border-zinc-200 hover:bg-transparent">
          {columns.map((column) => (
            <th
              key={column.key}
              className={cn("px-3 py-2 font-medium text-zinc-500", column.headerClassName)}
            >
              {column.header}
            </th>
          ))}
        </tr>
      </thead>
      <tbody className="divide-y divide-zinc-100">
        {rows.map((row, index) => (
          <tr
            key={getRowKey(row, index)}
            className={cn(
              "border-zinc-100 hover:bg-zinc-50",
              onRowClick && "cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400",
              rowClassName?.(row, index),
            )}
            tabIndex={onRowClick ? 0 : undefined}
            role={onRowClick ? "button" : undefined}
            onClick={() => onRowClick?.(row, index)}
            onKeyDown={(event) => handleRowKeyDown(event, row, index)}
          >
            {columns.map((column) => (
              <td key={column.key} className={cn("px-3 py-2", column.className)}>
                {column.cell(row, index)}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function ScoreMeter({ value }: { value?: number | null }) {
  const normalized = value == null ? 0 : value <= 1 ? value * 100 : Math.min(value, 100);
  return (
    <div className="flex min-w-28 items-center gap-2">
      <div className="h-2 w-20 overflow-hidden rounded-full bg-zinc-200">
        <div className="h-full rounded-full bg-emerald-500" style={{ width: `${normalized}%` }} />
      </div>
      <span className="w-12 text-right text-xs tabular-nums text-zinc-600">
        {value == null ? "n/a" : value <= 1 ? `${Math.round(value * 100)}%` : value.toFixed(1)}
      </span>
    </div>
  );
}

type ButtonProps = ComponentPropsWithoutRef<"button"> & {
  variant?: string;
};

export function Button({
  className,
  type = "button",
  variant: _variant,
  ...props
}: ButtonProps) {
  return (
    <button
      className={cn("inline-flex items-center justify-center", className)}
      type={type}
      {...props}
    />
  );
}

type CopyButtonProps = ButtonProps & {
  value: string;
  copyLabel?: string;
  copiedLabel?: string;
};

export function CopyButton({
  value,
  copyLabel = "Copy",
  copiedLabel = "Copied",
  className,
  onClick,
  ...props
}: CopyButtonProps) {
  const [label, setLabel] = useState(copyLabel);

  async function copyToClipboard(event: MouseEvent<HTMLButtonElement>) {
    onClick?.(event);
    if (event.defaultPrevented) {
      return;
    }
    try {
      await navigator.clipboard.writeText(value);
      setLabel(copiedLabel);
      window.setTimeout(() => setLabel(copyLabel), 1200);
    } catch {
      setLabel("Copy failed");
      window.setTimeout(() => setLabel(copyLabel), 1600);
    }
  }

  return (
    <Button className={className} onClick={(event) => void copyToClipboard(event)} {...props}>
      {label}
    </Button>
  );
}

export function FieldGroup({ className, ...props }: ComponentPropsWithoutRef<"div">) {
  return <div className={cn("grid", className)} {...props} />;
}

export function Field({ className, ...props }: ComponentPropsWithoutRef<"div">) {
  return <div className={cn("grid gap-2", className)} {...props} />;
}

export function FieldContent({ className, ...props }: ComponentPropsWithoutRef<"div">) {
  return <div className={cn("grid gap-1", className)} {...props} />;
}

export function FieldLabel({ className, ...props }: ComponentPropsWithoutRef<"label">) {
  return <label className={cn("text-sm font-medium text-zinc-950", className)} {...props} />;
}

export function FieldDescription({ className, ...props }: ComponentPropsWithoutRef<"p">) {
  return <p className={cn("text-sm leading-6 text-zinc-600", className)} {...props} />;
}

export function Input({ className, ...props }: ComponentPropsWithoutRef<"input">) {
  return (
    <input
      className={cn(
        "min-h-10 rounded-md border border-zinc-300 bg-white px-3 text-sm text-zinc-950 outline-none focus:border-zinc-500 focus:ring-2 focus:ring-zinc-200",
        className,
      )}
      {...props}
    />
  );
}

export function Textarea({ className, ...props }: ComponentPropsWithoutRef<"textarea">) {
  return (
    <textarea
      className={cn(
        "w-full rounded-md border border-zinc-300 bg-white p-3 text-sm text-zinc-950 outline-none focus:border-zinc-500 focus:ring-2 focus:ring-zinc-200",
        className,
      )}
      {...props}
    />
  );
}
